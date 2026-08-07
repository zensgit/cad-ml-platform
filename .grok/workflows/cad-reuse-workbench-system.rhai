// Master orchestration for CAD Reuse Workbench 90-day engineering closure.
// Coordinates gap scan, residual routing, and report for execute-plan handoff.
//
// args:
//   root  — repo root (default ".")
//   phase — "all" | "gap" | "verify" | "residual" (default "all")
//   pr547 — optional PR number string for status note (default "547")

let meta = #{
    name: "cad-reuse-workbench-system",
    description: "Master system workflow: gap inventory, verify Track R, residual E/S/O checklist, execute-plan handoff report",
    when_to_use: "When driving the full CAD_REUSE_WORKBENCH system of workflows + task board + execute-plan residuals",
    phases: [
        #{ title: "Inventory", detail: "read plan + task board + design" },
        #{ title: "Gap", detail: "parallel acceptance gaps" },
        #{ title: "VerifyR", detail: "Track R MVP verify" },
        #{ title: "Residual", detail: "E/S/O residual + human Track C" },
        #{ title: "Handoff", detail: "execute-plan instructions report" },
    ],
};

let summary_schema = #{
    "type": "object",
    "required": ["summary", "blockers", "next_actions"],
    "properties": #{
        "summary": #{ "type": "string" },
        "blockers": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "next_actions": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["ok", "evidence", "gaps"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "evidence": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 24 },
        "gaps": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 24 },
    },
};

let phase_arg = if args == () { "all" } else if args.phase == () { "all" } else { args.phase };
let root = if args == () { "." } else if args.root == () { "." } else { args.root };
let pr547 = if args == () { "547" } else if args.pr547 == () { "547" } else { args.pr547 };

phase("Inventory");
log("system workflow phase=" + phase_arg + " root=" + root);

let inv_p = "";
inv_p += "Read with tools at " + root + ": ";
inv_p += "docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md, ";
inv_p += "docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md if present, ";
inv_p += "docs/development/CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md if present, ";
inv_p += "docs/development/CAD_REUSE_WORKBENCH_MVP_VERIFICATION_20260808.md if present. ";
inv_p += "Summarize what is done vs residual_eng vs residual_human. ";
inv_p += "Mention L3 runtime WIP=1 and PR " + pr547 + " if relevant. ";
inv_p += "Return {summary, blockers, next_actions}.";

let inv = agent(inv_p, #{
    label: "inventory",
    capability_mode: "read-only",
    output_schema: summary_schema,
});

if inv == () || !inv.success {
    complete(#{ summary: "inventory failed", ok: false });
}

if phase_arg == "gap" || phase_arg == "all" {
    phase("Gap");
    let gj = [];
    gj.push(#{
        prompt: "At " + root + " verify Track R files exist: design-lock, src/core/review_reuse, api review_reuse, tests, decision default-off. Return {ok, evidence, gaps}.",
        label: "gap-r",
        capability_mode: "read-only",
        output_schema: verify_schema,
    });
    gj.push(#{
        prompt: "At " + root + " check residual audit docs TRACK_E_E1_POST_MERGE_AUDIT and TRACK_S_SEAL_IDENTITY_BASELINE_AUDIT and TRACK_O_PILOT_OPS_PACKAGE. ok=true only if all three exist with substance. Return {ok, evidence, gaps}.",
        label: "gap-eso",
        capability_mode: "read-only",
        output_schema: verify_schema,
    });
    let gres = parallel(gj);
    log("gap panel done");
}

let r_ok = false;
if phase_arg == "verify" || phase_arg == "all" {
    phase("VerifyR");
    let vp = "";
    vp += "Verify Track R MVP at " + root + " with tools. Require EvidencePack §3.3 fields in code, decision default-off, tenant isolation tests present. ";
    vp += "Set ok=true only with file evidence. Return {ok, evidence, gaps}.";
    let vr = agent(vp, #{
        label: "verify-r",
        capability_mode: "read-only",
        output_schema: verify_schema,
    });
    r_ok = vr != () && vr.success && vr.output.ok == true;
}

let residual_out = ();
if phase_arg == "residual" || phase_arg == "all" {
    phase("Residual");
    let rp = "";
    rp += "Produce residual checklist for " + root + " workbench system. ";
    rp += "Split residual_eng (E2/S3/O2/R10/O3) vs residual_human (Track C, owner ratify, pilot enable). ";
    rp += "Recommend execute-plan batch: PR1+PR2 now; PR3+PR4 after PR " + pr547 + " L3 slot free. ";
    rp += "Return {summary, blockers, next_actions}.";
    residual_out = agent(rp, #{
        label: "residual",
        capability_mode: "read-only",
        output_schema: summary_schema,
    });
}

phase("Handoff");
let report = "# CAD Reuse Workbench SYSTEM report\n\n";
report += "phase=" + phase_arg + "\nroot=" + root + "\npr=" + pr547 + "\n\n";
report += "## Inventory\n" + inv.output.summary + "\n\n";
report += "blockers: " + json_encode(inv.output.blockers) + "\n";
report += "next: " + json_encode(inv.output.next_actions) + "\n\n";
report += "Track R verify ok=" + r_ok.to_string() + "\n\n";
if residual_out != () && residual_out.success {
    report += "## Residual\n" + residual_out.output.summary + "\n";
    report += "next: " + json_encode(residual_out.output.next_actions) + "\n\n";
}
report += "## execute-plan handoff\n";
report += "Design: docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md\n";
report += "Command: /execute-plan docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md --no-graphite --auto-pr --concurrency 2\n";
report += "Instructions: L3 WIP=1; no eval_integrity replace; no cost_cap; decision default-off; Track C residual-human only.\n";
report += "First batch while #" + pr547 + " open: PR1 + PR2 only.\n";

let path = write_scratch_file("workbench_system_report.md", report);
complete(#{
    summary: "system orchestration report ready",
    phase: phase_arg,
    ok: r_ok || phase_arg == "gap" || phase_arg == "residual",
    track_r_ok: r_ok,
    report_path: path,
    inventory: inv.output,
    residual: if residual_out != () && residual_out.success { residual_out.output } else { () },
    execute_plan_design: "docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md",
});
