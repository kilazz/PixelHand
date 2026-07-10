// build.rs

fn main() {
    // Instruct Cargo to re-run this build script only if the 'ui' directory changes.
    // This prevents redundant recompilations of the Slint UI when only Rust code is modified.
    println!("cargo:rerun-if-changed=ui");

    // Compile the declarative Slint markup file.
    // Cargo runs the build script with the workspace root directory as the current working directory,
    // so we can reference "ui/app.slint" directly.
    slint_build::compile("ui/app.slint").unwrap();
}
