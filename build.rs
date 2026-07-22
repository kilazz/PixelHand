// build.rs

fn main() {
    // Instruct Cargo to re-run this build script if the 'ui' directory or PixelHand.rc changes.
    println!("cargo:rerun-if-changed=ui");
    println!("cargo:rerun-if-changed=PixelHand.rc");

    // Compile the declarative Slint markup file.
    slint_build::compile("ui/app.slint").unwrap();

    // Embed Windows resources (icon, metadata) during compilation on Windows.
    #[cfg(target_os = "windows")]
    {
        embed_resource::compile("PixelHand.rc", embed_resource::NONE)
            .manifest_optional()
            .unwrap();
    }
}
