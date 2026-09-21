"""Numerical Shepp--Logan reconstruction with the paper's nominal Sine Spin protocols.

Forward and backprojection use LEAP Joseph kernels for BOTH orbits. Reconstructions use
nonnegative LS from zeros, with no FDK calibration, support clipping or post-hoc
image intensity rescaling. Finite-detector visibility is reported separately from the
reconstructed usable region and the paper's proprietary Grangeat support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from shepp_logan import create_shepp_logan, phantom_metadata
from sinespin_geometry import build_icono_orbit


PAPER_URL = "https://doi.org/10.1117/1.JMI.11.4.043503"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def geometry_record(geo):
    return dict(
        kind=geo.kind, views=geo.n_views, scan_angle_deg=geo.scan_angle_deg,
        angle_step_deg=float(geo.theta_deg[1] - geo.theta_deg[0]),
        tilt_min_max_deg=[float(geo.tilt_deg.min()), float(geo.tilt_deg.max())],
        tilt_cycles=0 if geo.kind == "circular" else 1,
        sod_mm=geo.sod_mm, sdd_mm=geo.sdd_mm,
        detector_shape_vu=[geo.detector_rows, geo.detector_cols],
        pixel_vu_mm=[geo.pixel_height, geo.pixel_width],
        detector_extent_vu_mm=[geo.detector_height_mm, geo.detector_width_mm],
        principal_point_uv=[(geo.detector_cols-1)/2, (geo.detector_rows-1)/2],
        isocenter_xyz_mm=[0, 0, 0], include_endpoints=True,
        distances_are_assumed=True,
    )


def configure_leap(geo, shape, voxel, device):
    from leapctype import tomographicModels

    ct = tomographicModels()
    ct.set_gpu(device.index or 0)
    ct.print_warnings = False
    ct.print_cost = False
    if not hasattr(ct, "get_forceJosephModularBackprojection"):
        raise RuntimeError("This experiment requires the Joseph-pinned LEAP build; see docs/sinespin.md.")
    if not ct.set_forceJosephModular(True) or not ct.get_forceJosephModular():
        raise RuntimeError("LEAP could not pin the forward projector to Joseph.")
    if not ct.get_forceJosephModularBackprojection():
        raise RuntimeError("LEAP could not pin the backprojector to Joseph.")
    if not ct.set_modularbeam(geo.n_views, geo.detector_rows, geo.detector_cols,
                             geo.pixel_height, geo.pixel_width, *geo.modular_arrays()):
        raise RuntimeError("LEAP rejected the detector geometry.")
    nz, ny, nx = shape
    if not ct.set_volume(nx, ny, nz, voxel, voxel, 0.0, 0.0, 0.0):
        raise RuntimeError("LEAP rejected the centered reconstruction grid.")
    ct.set_volumeDimensionOrder(1)  # LEAP ZYX
    ct.set_diameterFOV(float(np.hypot(nx, ny) * voxel))
    ct.set_offsetScan(False)
    ct.set_truncatedScan(False)
    return ct


def metrics_torch(recon, gt, masks):
    import torch

    result = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            result[name] = None
            continue
        truth = gt[mask]
        error = recon[mask] - truth
        result[name] = dict(
            voxel_count=int(mask.sum()),
            rel_rmse=float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(truth).clamp_min(1e-12)),
            mean_bias=float(error.mean()),
        )
    return result


def central_interval(z, good):
    center = int(np.argmin(np.abs(z)))
    if not good[center]:
        return None
    lo = hi = center
    while lo > 0 and good[lo-1]:
        lo -= 1
    while hi+1 < len(z) and good[hi+1]:
        hi += 1
    step = float(z[1] - z[0])
    return dict(lower_mm=float(z[lo]-step/2), upper_mm=float(z[hi]+step/2),
                height_mm=float(z[hi]-z[lo]+step))


def reconstruction_profiles(recon, gt, xyz, radius=65.0):
    x, y, z = xyz
    X, Y = np.meshgrid(x, y, indexing="xy")
    profiles = {}
    for label, px, py in (("x+65", radius, 0), ("x-65", -radius, 0),
                          ("y+65", 0, radius), ("y-65", 0, -radius)):
        patch = (X-px)**2 + (Y-py)**2 <= 3.0**2
        t, r = gt[:, patch], recon[:, patch]
        gt_mean = t.mean(1)
        rec_mean = r.mean(1)
        rmse = np.sqrt(((r-t)**2).mean(1)) / np.maximum(np.sqrt((t*t).mean(1)), 1e-8)
        tissue = gt_mean >= .002
        profiles[label] = dict(
            xy_mm=[px, py], z_mm=z.tolist(), gt_mean=gt_mean.tolist(),
            recon_mean=rec_mean.tolist(), relative_rmse=rmse.tolist(),
            patch_radius_mm=3.0,
            central_usable_intervals={str(tol): central_interval(z, tissue & (rmse <= tol))
                                      for tol in (.05, .1, .2)},
        )
    return profiles


def make_figures(out, geometries, xyz, gt=None, recons=None, profiles=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = xyz
    X, Z = np.meshgrid(x, z, indexing="xy")
    # Sagittal section y=0. Geometry uses the exact plane, image uses nearest slice.
    points = np.stack((X, np.zeros_like(X), Z), axis=-1)
    fractions = {name: geo.visibility_fraction(points, chunk_size=2048)
                 for name, geo in geometries.items()}
    bounds = {name: geo.longitudinal_intervals(np.stack((x, np.zeros_like(x)), -1))
              for name, geo in geometries.items()}
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    for name, geo in geometries.items():
        color = "#1565c0" if name == "circular" else "#dd8500"
        s, d = geo.source_positions, geo.module_centers
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=color, label=name+" source")
        ax.plot(d[:, 0], d[:, 1], d[:, 2], color=color, ls="--", alpha=.6)
        for i in (0, geo.n_views//4, geo.n_views//2):
            ax.plot([s[i,0],d[i,0]], [s[i,1],d[i,1]], [s[i,2],d[i,2]], color=color, alpha=.2)
    ax.scatter([0], [0], [0], color="k", s=15)
    orbit_points=np.concatenate([points for geo in geometries.values()
                                 for points in (geo.source_positions,geo.module_centers)])
    ax.set_box_aspect(np.maximum(np.ptp(orbit_points,axis=0),1.0))
    ax.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title="Source and detector: common isocenter")
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(122)
    for name, geo in geometries.items():
        ax2.plot(geo.theta_deg, geo.tilt_deg, label=f"{name}: {geo.n_views} views")
    ax2.set(xlabel="Azimuth [deg]", ylabel="Tilt [deg]", title="Paper arcs; one sine cycle over 220 deg")
    ax2.legend(); ax2.grid(alpha=.2)
    fig.savefig(out/"orbit.png", dpi=160); plt.close(fig)

    extent = [x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2,
              z[0]-(z[1]-z[0])/2, z[-1]+(z[1]-z[0])/2]
    if gt is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        for ax, (name, fraction) in zip(axes, fractions.items()):
            im = ax.imshow(fraction, extent=extent, origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax.set(title=name, xlabel="x [mm]", ylabel="z [mm]")
        fig.colorbar(im, ax=axes, label="Fraction of views inside detector")
        fig.savefig(out/"detector_visibility.png", dpi=160); plt.close(fig)
        return
    yi = int(np.argmin(np.abs(y)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    image_rows = [("Ground truth", gt), ("Circular / Joseph LS", recons["circular"]),
                  ("Sine Spin / Joseph LS", recons["sinespin"])]
    for ax, (title, volume) in zip(axes[0], image_rows):
        ax.imshow(volume[:, yi, :], extent=extent, origin="lower", cmap="gray", vmin=0, vmax=.006)
        ax.set(title=title, xlabel="x [mm]", ylabel="z [mm]", xlim=(-110,110), ylim=(-140,140))
        for xp in (-65, 65): ax.axvline(xp, color="#44ccee", lw=.6, alpha=.6)
    for j, (name, geo) in enumerate(geometries.items()):
        ax = axes[1,j]
        im = ax.imshow(fractions[name], extent=extent, origin="lower", cmap="viridis", vmin=0, vmax=1)
        for k in (0,1): ax.plot(x, bounds[name][:,k], color="white", lw=1.1)
        ax.axvline(65, color="#ff5599", ls="--", lw=.8)
        interval = geo.longitudinal_intervals(np.array([[65.,0.]]))[0]
        ax.plot([65,65], interval, color="#ff5599", lw=2)
        ax.set(title=f"{name}: all-view visibility {np.diff(interval)[0]:.1f} mm at x=65",
               xlabel="x [mm]", ylabel="z [mm]", xlim=(-110,110), ylim=(-140,140))
    fig.colorbar(im, ax=axes[1,:2], shrink=.7, label="View fraction (not a reconstruction mask)")
    ax = axes[1,2]
    p0=profiles["circular"]["x+65"]
    ax.plot(p0["z_mm"],p0["gt_mean"],color="black",label="GT")
    for name in geometries:
        p=profiles[name]["x+65"]
        ax.plot(p["z_mm"],p["recon_mean"],label=name,lw=1.5)
    ax.set(xlabel="z [mm] at x=65, y=0", ylabel="Mean attenuation [1/mm]",
           title="Raw reconstruction profile; no FOV clipping", xlim=(-130,130))
    ax.legend(); ax.grid(alpha=.2)
    nominal=geometries["sinespin"]
    fig.suptitle(f"Shepp–Logan FOV test | {nominal.sod_mm:g}/{nominal.sdd_mm:g} mm assumed distances | raw numerical reconstructions",fontsize=12)
    fig.savefig(out/"fig3_fov_comparison.png",dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,label in zip(axes.flat,profiles["circular"]):
        for name in geometries:
            p=profiles[name][label]
            ax.plot(p["z_mm"],p["relative_rmse"],label=name)
        ax.axhline(.1,color="black",ls="--",lw=.8,label="10% local error")
        ax.set(title=label+" mm, 3-mm patch",xlabel="z [mm]",ylabel="Local relative RMSE",ylim=(0,.6))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/"reconstruction_fov_profiles.png",dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=Path("result_sinespin/shepp_logan_fig3"))
    parser.add_argument("--voxel-mm",type=float,default=1.0)
    parser.add_argument("--volume-mm",type=float,default=280.0)
    parser.add_argument("--detector-bin",type=int,default=2,help="Additional numerical downsampling; detector footprint preserved.")
    parser.add_argument("--iterations",type=int,default=160)
    parser.add_argument("--check-every",type=int,default=40)
    parser.add_argument("--gpu",type=int,default=1,help="Physical GPU index; defaults to the assigned GPU 1.")
    parser.add_argument("--sod-mm",type=float,default=750.0)
    parser.add_argument("--sdd-mm",type=float,default=1200.0)
    parser.add_argument("--geometry-only",action="store_true")
    parser.add_argument("--resume",action="store_true",help="Continue both saved reconstructions to --iterations in the same output directory.")
    args=parser.parse_args()
    if min(args.voxel_mm,args.volume_mm,args.iterations,args.check_every,args.detector_bin)<=0:
        parser.error("Grid, iterations, check interval and detector bin must be positive.")
    if args.volume_mm < 260:
        parser.error("The 260-mm phantom must fit without cropping; choose volume-mm >=260.")
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    n=int(np.ceil(args.volume_mm/args.voxel_mm)); shape=(n,n,n)
    axis=(np.arange(n)-(n-1)/2)*args.voxel_mm; xyz=(axis,axis,axis)
    geometries={name:build_icono_orbit(name,detector_bin=args.detector_bin,sod_mm=args.sod_mm,sdd_mm=args.sdd_mm)
                for name in ("circular","sinespin")}
    records={name:geometry_record(geo) for name,geo in geometries.items()}
    fov={name:geo.fov_at_radius(65) for name,geo in geometries.items()}
    run=dict(paper=PAPER_URL,geometry=records,finite_detector_fov_at_radius65mm=fov,
             grid=dict(shape_zyx=list(shape),voxel_mm=args.voxel_mm,origin="isocenter-centered"),
             phantom=phantom_metadata(),reconstruction=dict(method="LEAP nonnegative LS (SQS)",
                 forward_projector="Joseph, forceJosephModular=True",initialization="zeros",iterations=args.iterations,
                 backprojector="LEAP Joseph modular kernel, forceJosephModularBackprojection=True; approximate adjoint",
                 restart_block=args.check_every,support_mask_applied=False,posthoc_intensity_rescaling_applied=False),
             paper_fig3_target_mm=dict(circular=160,sinespin=120),
             caveat="Distances/calibrated poses and proprietary Grangeat reconstruction are not supplied by the paper.")
    previous = None
    if args.resume:
        if args.geometry_only:
            parser.error("--resume cannot be combined with --geometry-only")
        previous = json.loads((out/"metrics.json").read_text())
        for key in ("geometry", "grid", "phantom"):
            if previous[key] != run[key]:
                parser.error(f"Saved {key} differs from the requested experiment")
        if previous["reconstruction"]["restart_block"] != args.check_every:
            parser.error("Keep the same --check-every when resuming")
        if previous["reconstruction"].get("backprojector") != run["reconstruction"]["backprojector"]:
            parser.error("Saved run uses a different backprojector; start a new experiment")
        for name in geometries:
            if not (out/("recon_"+name+".npy")).exists():
                parser.error(f"Missing saved {name} reconstruction")
            if previous["results"][name]["convergence"][-1]["iterations"] > args.iterations:
                parser.error("--iterations must be at least the saved iteration count")
    save_json(out/"geometry.json",run)
    for name,geo in geometries.items():
        np.savez(out/(name+"_geometry.npz"),source_positions=geo.source_positions,module_centers=geo.module_centers,
                 row_vectors=geo.row_vectors,col_vectors=geo.col_vectors,P_pixel=geo.projection_matrices(),
                 theta_deg=geo.theta_deg,tilt_deg=geo.tilt_deg)
        print(f"[geometry] {name}: {geo.n_views} views/{geo.scan_angle_deg} deg; "
              f"r65 all-view FOV {fov[name]['height_min_mm']:.2f}..{fov[name]['height_max_mm']:.2f} mm",flush=True)
    if args.geometry_only:
        make_figures(out,geometries,xyz)
        return

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("The requested NVIDIA CUDA GPU is unavailable.")
    device=torch.device("cuda:0"); torch.cuda.set_device(device)
    run["runtime"]=dict(physical_gpu=args.gpu,gpu_name=torch.cuda.get_device_name(device),
                        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__)
    print(f"[device] physical GPU {args.gpu}: {torch.cuda.get_device_name(device)}",flush=True)
    gt=create_shepp_logan(shape,args.voxel_mm)
    np.save(out/"ground_truth.npy",gt)
    gt_gpu=torch.from_numpy(gt).to(device)
    X,Y=np.meshgrid(axis,axis,indexing="xy")
    xy=np.stack((X,Y),axis=-1)
    masks={}
    for name,geo in geometries.items():
        limits=geo.longitudinal_intervals(xy)
        masks[name]=(axis[:,None,None]>=limits[None,:,:,0]) & (axis[:,None,None]<=limits[None,:,:,1])
    common=torch.from_numpy(masks["circular"] & masks["sinespin"]).to(device)
    foreground=gt_gpu>0
    metric_masks=dict(object=foreground,common_visible_object=foreground & common,
                      not_common_visible_object=foreground & ~common)
    recons={};profiles={};run["results"]={}
    for name,geo in geometries.items():
        ct=configure_leap(geo,shape,args.voxel_mm,device)
        library=Path(ct.libprojectors._name).resolve()
        run["runtime"]["leap_library_sha256"]=hashlib.sha256(library.read_bytes()).hexdigest()
        if previous is not None and previous.get("runtime",{}).get("leap_library_sha256") != run["runtime"]["leap_library_sha256"]:
            raise RuntimeError("Saved run uses a different LEAP library; start a new experiment")
        g=torch.zeros((geo.n_views,geo.detector_rows,geo.detector_cols),dtype=torch.float32,device=device)
        torch.cuda.synchronize();start=time.perf_counter()
        ct.project_gpu(g,gt_gpu)
        torch.cuda.synchronize();project_seconds=time.perf_counter()-start
        if not bool(torch.isfinite(g).all()) or float(g.abs().max())==0:
            raise RuntimeError("Invalid projections from LEAP.")
        gnorm=torch.linalg.vector_norm(g)
        if previous is None:
            f=torch.zeros_like(gt_gpu); history=[]; done=0; prior_seconds=0.0
        else:
            f=torch.from_numpy(np.load(out/("recon_"+name+".npy"))).to(device)
            if f.shape != gt_gpu.shape or f.dtype != torch.float32 or not bool(torch.isfinite(f).all()):
                raise RuntimeError(f"Invalid saved {name} reconstruction")
            history=previous["results"][name]["convergence"].copy()
            history[-1].setdefault("usable_fov", previous["results"][name]["usable_fov"])
            done=history[-1]["iterations"]; prior_seconds=history[-1]["elapsed_seconds"]
        projected=torch.empty_like(g); start=time.perf_counter()
        print(f"[{name}] Joseph projection {project_seconds:.2f}s; LS iterations {done} -> {args.iterations}",flush=True)
        while done<args.iterations:
            block=min(args.check_every,args.iterations-done)
            if ct.LS(g,f,block,"SQS",True) is None: raise RuntimeError("LEAP LS failed.")
            done+=block
            ct.project_gpu(projected,f)
            residual=float(torch.linalg.vector_norm(projected-g)/gnorm)
            if not bool(torch.isfinite(f).all()): raise RuntimeError("Nonfinite reconstruction.")
            measure=metrics_torch(f,gt_gpu,metric_masks)
            checkpoint_profiles=reconstruction_profiles(f.cpu().numpy(),gt,xyz)
            torch.cuda.synchronize()
            history.append(dict(iterations=done,relative_projection_residual=residual,
                                reconstruction=measure,elapsed_seconds=prior_seconds+time.perf_counter()-start,
                                usable_fov={key:p["central_usable_intervals"] for key,p in checkpoint_profiles.items()}))
            save_json(out/(name+"_convergence.json"),history)
            print(f"[{name}] iter {done}: projection residual={residual:.5g}, "
                  f"common-object relRMSE={measure['common_visible_object']['rel_rmse']:.5g}",flush=True)
        rec=f.cpu().numpy();np.save(out/("recon_"+name+".npy"),rec)
        recons[name]=rec;profiles[name]=reconstruction_profiles(rec,gt,xyz)
        run["results"][name]=dict(projection_seconds=project_seconds,convergence=history,
                                   usable_fov={key:p["central_usable_intervals"] for key,p in profiles[name].items()})
        del ct,g,f,projected;torch.cuda.empty_cache()
        save_json(out/"metrics.json",run)
    save_json(out/"profiles.json",profiles)
    run["peak_torch_gpu_memory_GiB"]=torch.cuda.max_memory_allocated()/2**30
    save_json(out/"metrics.json",run)
    make_figures(out,geometries,xyz,gt,recons,profiles)
    print(f"[done] {out.resolve()}",flush=True)


if __name__ == "__main__":
    main()
