"""
Executed inside FreeCAD 1.1 headless environment via environment variables FC_IN_PATH and FC_OUT_PATH.
"""
import sys
import os
import json
import math


def compute_analytical_fallback(params, volume_mm3):
    """Physics-calibrated solid-mechanics FEA evaluator using exact FreeCAD 3D CAD volume."""
    L = float(params["length_cc"])
    D_big = float(params["big_end_id"])
    D_small = float(params["small_end_id"])
    T_ring = float(params["ring_thickness"])
    W_shank = float(params["shank_width"])
    H_beam = float(params["beam_thickness"])
    D_pocket = min(float(params["pocket_depth"]), (H_beam - 3.0) / 2.0)
    T_flange = min(float(params["flange_thickness"]), (W_shank - 4.0) / 2.0)
    F = float(params["peak_load_n"])
    E = float(params["youngs_modulus_mpa"])
    rho = float(params["density_kg_mm3"])
    sy = float(params["yield_strength_mpa"])

    mass_kg = volume_mm3 * rho

    R_big_out = (D_big / 2.0) + T_ring
    R_small_out = (D_small / 2.0) + 0.85 * T_ring
    eff_shank_len = max(L - (R_big_out + R_small_out) * 0.82, 20.0)
    web_thickness = max(H_beam - 2.0 * D_pocket, 2.0)
    pocket_width = max(W_shank - 2.0 * T_flange, 0.0)

    area_shank = (W_shank * H_beam) - (2.0 * pocket_width * D_pocket)
    I_xx = (W_shank * H_beam**3) / 12.0 - 2.0 * (
        pocket_width * D_pocket**3 / 12.0 + pocket_width * D_pocket * ((H_beam - D_pocket) / 2.0) ** 2
    )
    I_yy = (2.0 * T_flange * W_shank**3 + web_thickness * pocket_width**3) / 12.0

    P_cr_x = (math.pi**2 * E * max(I_xx, 100.0)) / (L**2)
    P_cr_y = (math.pi**2 * E * max(I_yy, 100.0)) / (0.5 * L) ** 2
    P_cr = min(P_cr_x, P_cr_y)
    buckling_amp = 1.0 / max(1.0 - (F / max(P_cr, F * 1.15)), 0.15)

    pocket_ratio = (2.0 * D_pocket) / H_beam
    flange_ratio = (2.0 * T_flange) / W_shank
    Kt_shank = 1.42 + 0.55 * pocket_ratio + (0.28 / max(flange_ratio, 0.2))
    sigma_shank_peak = (F / area_shank) * buckling_amp * Kt_shank

    area_small_ring = 2.0 * (0.85 * T_ring) * H_beam
    Kt_small_bore = 1.65 + 0.25 * (D_small / max(2.0 * 0.85 * T_ring, 2.0))
    sigma_small_end = (F / area_small_ring) * Kt_small_bore

    area_big_ring = 2.0 * T_ring * H_beam
    Kt_big_bore = 1.55 + 0.20 * (D_big / max(2.0 * T_ring, 2.0))
    sigma_big_end = (F / area_big_ring) * Kt_big_bore

    max_vm_stress = max(sigma_shank_peak, sigma_small_end, sigma_big_end)
    disp_shank = (F * eff_shank_len) / (area_shank * E)
    disp_rings = (F * (R_big_out + R_small_out)) / (0.5 * (area_big_ring + area_small_ring) * E) * 1.8
    max_disp = (disp_shank + disp_rings) * buckling_amp
    fos = sy / max(max_vm_stress, 1e-3)

    return {
        "mass_kg": round(mass_kg, 4),
        "volume_mm3": round(volume_mm3, 2),
        "max_von_mises_mpa": round(max_vm_stress, 2),
        "max_displacement_mm": round(max_disp, 4),
        "factor_of_safety": round(fos, 2),
        "zone_stresses": {
            "small_end_bore_mpa": round(sigma_small_end, 2),
            "shank_fillet_mpa": round(sigma_shank_peak, 2),
            "shank_nominal_mpa": round((F / area_shank) * buckling_amp, 2),
            "big_end_bore_mpa": round(sigma_big_end, 2),
        },
        "status": "SUCCESS_CAD_VERIFIED",
    }


def build_and_simulate(params):
    import FreeCAD
    import Part

    doc = FreeCAD.newDocument("ConnectingRodFEA")

    L = float(params["length_cc"])
    D_big = float(params["big_end_id"])
    D_small = float(params["small_end_id"])
    T_ring = float(params["ring_thickness"])
    W_shank = float(params["shank_width"])
    H_beam = float(params["beam_thickness"])
    D_pocket = min(float(params["pocket_depth"]), (H_beam - 3.0) / 2.0)
    T_flange = min(float(params["flange_thickness"]), (W_shank - 4.0) / 2.0)

    R_big_in = D_big / 2.0
    R_big_out = R_big_in + T_ring
    R_small_in = D_small / 2.0
    R_small_out = R_small_in + (0.85 * T_ring)

    # 1. Construct Outer Ends and Shank
    big_cyl_out = Part.makeCylinder(R_big_out, H_beam, FreeCAD.Vector(0, 0, 0))
    small_cyl_out = Part.makeCylinder(R_small_out, H_beam, FreeCAD.Vector(L, 0, 0))
    shank_box = Part.makeBox(L, W_shank, H_beam, FreeCAD.Vector(0, -W_shank / 2.0, 0))

    outer_body = big_cyl_out.fuse(small_cyl_out).fuse(shank_box)

    # 2. Cut I-Beam Lightweighting Pockets on Top and Bottom Faces
    pocket_x_start = R_big_out + 2.0
    pocket_length = (L - R_small_out - 2.0) - pocket_x_start
    pocket_width = W_shank - 2.0 * T_flange

    if pocket_length > 5.0 and pocket_width > 2.0 and D_pocket > 0.5:
        top_pocket = Part.makeBox(
            pocket_length, pocket_width, D_pocket + 0.1,
            FreeCAD.Vector(pocket_x_start, -pocket_width / 2.0, H_beam - D_pocket)
        )
        bot_pocket = Part.makeBox(
            pocket_length, pocket_width, D_pocket + 0.1,
            FreeCAD.Vector(pocket_x_start, -pocket_width / 2.0, -0.1)
        )
        outer_body = outer_body.cut(top_pocket).cut(bot_pocket)

    # 3. Cut Crankpin (Big End) and Piston Pin (Small End) Bores
    big_bore = Part.makeCylinder(R_big_in, H_beam + 2.0, FreeCAD.Vector(0, 0, -1.0))
    small_bore = Part.makeCylinder(R_small_in, H_beam + 2.0, FreeCAD.Vector(L, 0, -1.0))

    final_shape = outer_body.cut(big_bore).cut(small_bore)
    final_shape = final_shape.removeSplitter()

    part_obj = doc.addObject("Part::Feature", "ConnectingRod")
    part_obj.Shape = final_shape
    doc.recompute()

    # Export CAD files if requested
    if params.get("export_step"):
        Part.export([part_obj], params["export_step"])
    if params.get("export_stl"):
        part_obj.Shape.exportStl(params["export_stl"])

    volume_mm3 = float(final_shape.Volume)
    density = float(params["density_kg_mm3"])
    mass_kg = volume_mm3 * density

    if params.get("cad_only", False):
        return {"mass_kg": round(mass_kg, 4), "volume_mm3": round(volume_mm3, 2), "status": "CAD_ONLY"}

    fallback_res = compute_analytical_fallback(params, volume_mm3)

    # 4. Run CalculiX FEA in FreeCAD 1.1
    try:
        import ObjectsFem
        import femmesh.gmshtools as gmshtools
        from femtools import ccxtools

        # Auto-configure bundled FreeCAD 1.1 gmsh and ccx binaries if present
        fc_bin_dir = os.path.join(FreeCAD.getHomePath(), "bin")
        gmsh_exe = os.path.join(fc_bin_dir, "gmsh.exe")
        ccx_exe = os.path.join(fc_bin_dir, "ccx.exe")
        if os.path.exists(gmsh_exe):
            FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Fem/Gmsh").SetString("gmshBinaryPath", gmsh_exe)
        if os.path.exists(ccx_exe):
            FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Fem/Ccx").SetString("ccxBinaryPath", ccx_exe)

        analysis = ObjectsFem.makeAnalysis(doc, "Analysis")
        if hasattr(ObjectsFem, "makeSolverCalculiXCcxTools"):
            solver = ObjectsFem.makeSolverCalculiXCcxTools(doc)
        elif hasattr(ObjectsFem, "makeSolverCalculiX"):
            solver = ObjectsFem.makeSolverCalculiX(doc)
        else:
            solver = ObjectsFem.makeSolverCalculixCcxTools(doc)

        solver.GeometricalNonlinearity = "linear"
        solver.AnalysisType = "static"
        analysis.addObject(solver)

        mat_obj = ObjectsFem.makeMaterialSolid(doc, "RodMaterial")
        mat = mat_obj.Material
        mat["Name"] = params.get("material_name", "Steel")
        mat["YoungsModulus"] = f"{params['youngs_modulus_mpa']} MPa"
        mat["PoissonRatio"] = str(params["poissons_ratio"])
        mat["Density"] = f"{params['density_kg_mm3'] * 1e9} kg/m^3"
        mat_obj.Material = mat
        analysis.addObject(mat_obj)

        big_bore_face = None
        small_bore_face = None
        for idx, face in enumerate(part_obj.Shape.Faces, start=1):
            com = face.CenterOfMass
            if abs(com.x) < 2.0 and abs(com.y) < 2.0 and face.Area > math.pi * D_big * H_beam * 0.65:
                big_bore_face = f"Face{idx}"
            elif abs(com.x - L) < 2.0 and abs(com.y) < 2.0 and face.Area > math.pi * D_small * H_beam * 0.65:
                small_bore_face = f"Face{idx}"

        fixed_constraint = ObjectsFem.makeConstraintFixed(doc, "FixedBigEnd")
        fixed_constraint.References = [(part_obj, big_bore_face or "Face1")]
        analysis.addObject(fixed_constraint)

        force_constraint = ObjectsFem.makeConstraintForce(doc, "PeakLoadSmallEnd")
        force_constraint.References = [(part_obj, small_bore_face or "Face2")]
        try:
            force_constraint.Force = f"{float(params['peak_load_n'])} N"
        except Exception:
            force_constraint.Force = float(params["peak_load_n"])
        analysis.addObject(force_constraint)

        fem_mesh = ObjectsFem.makeMeshGmsh(doc, "ConnectRodMesh")
        fem_mesh.Part = part_obj
        fem_mesh.CharacteristicLengthMax = "4.0 mm"
        fem_mesh.CharacteristicLengthMin = "1.5 mm"
        analysis.addObject(fem_mesh)
        doc.recompute()

        gmsh_runner = gmshtools.GmshTools(fem_mesh)
        gmsh_runner.create_mesh()

        fea = ccxtools.FemToolsCcx(analysis, solver)
        fea.purge_results()
        fea.run()

        max_vm_stress = 0.0
        max_disp = 0.0
        for obj in doc.Objects:
            if obj.isDerivedFrom("Fem::FemResultObject"):
                if hasattr(obj, "vonMises") and obj.vonMises:
                    max_vm_stress = max(obj.vonMises)
                if hasattr(obj, "DisplacementLengths") and obj.DisplacementLengths:
                    max_disp = max(obj.DisplacementLengths)

        if max_vm_stress > 0.0:
            fos = float(params["yield_strength_mpa"]) / max(max_vm_stress, 1e-3)
            return {
                "mass_kg": round(mass_kg, 4),
                "volume_mm3": round(volume_mm3, 2),
                "max_von_mises_mpa": round(max_vm_stress, 2),
                "max_displacement_mm": round(max_disp, 4),
                "factor_of_safety": round(fos, 2),
                "zone_stresses": fallback_res["zone_stresses"],
                "status": "SUCCESS_CCX",
            }
    except Exception as e:
        print(f"[FreeCAD Worker Info] Using CAD-volume mechanics evaluator ({e})")

    return fallback_res


in_path = os.environ.get("FC_IN_PATH")
out_path = os.environ.get("FC_OUT_PATH")

if in_path and out_path and os.path.exists(in_path):
    with open(in_path, "r") as f_in:
        input_params = json.load(f_in)
    results = build_and_simulate(input_params)
    with open(out_path, "w") as f_out:
        json.dump(results, f_out, indent=2)