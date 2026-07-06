fn main() {
    // Compile the Slint markup file.
    // Cargo runs the build script with the workspace root directory as the current working directory,
    // so we reference "ui/app.slint" directly.
    slint_build::compile("ui/app.slint").unwrap();
}
