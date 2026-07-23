// src/reporting/html_report.rs

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::state::models::{DuplicateFileSummary, DuplicateGroupSummary, QcIssueSummary};

/// Escapes special characters for safe HTML output rendering.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Generates a standalone, print-ready HTML/PDF report with full asset metadata and CSS styling.
pub fn generate_html_report(
    groups: &[DuplicateGroupSummary],
    qc_issues: &[QcIssueSummary],
    inventory: &[DuplicateFileSummary],
    out_path: &Path,
) -> Result<()> {
    let mut html = String::with_capacity(1024 * 32);

    let total_files: usize = if !groups.is_empty() {
        groups.iter().map(|g| g.files.len()).sum()
    } else if !qc_issues.is_empty() {
        qc_issues.len()
    } else {
        inventory.len()
    };

    let total_wasted_bytes: u64 = groups
        .iter()
        .map(|g| {
            if g.files.len() > 1 {
                g.files.iter().skip(1).map(|f| f.size).sum::<u64>()
            } else {
                0
            }
        })
        .sum();

    let wasted_str = crate::utils::helpers::format_size(total_wasted_bytes);

    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PixelHand Audit Report</title>
    <style>
        :root {
            --bg-color: #1a1a1d;
            --card-bg: #2d2d32;
            --accent-color: #3e6a90;
            --text-primary: #f0f0f0;
            --text-secondary: #aaaaaa;
            --border-color: #3d3d42;
            --highlight: #f0b132;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: var(--card-bg);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--highlight);
        }

        .group-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 24px;
            overflow: hidden;
        }

        .group-header {
            background: #232328;
            padding: 12px 16px;
            font-weight: bold;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }

        th {
            background: #202024;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .best-badge {
            background: var(--highlight);
            color: #000;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 6px;
        }

        .path-cell {
            word-break: break-all;
            color: var(--text-secondary);
            font-family: monospace;
            font-size: 12px;
        }

        @media print {
            body {
                background-color: #ffffff;
                color: #000000;
            }
            .group-card, .stat-card {
                background: #ffffff;
                border: 1px solid #ccc;
                page-break-inside: avoid;
            }
            th {
                background: #f0f0f0;
                color: #333;
            }
            .stat-value {
                color: #000;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 28px;">PixelHand Technical Audit Report</h1>
                <p style="margin: 4px 0 0 0; color: var(--text-secondary);">Generated automatically via PixelHand AI Engine</p>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div style="font-size: 12px; color: var(--text-secondary);">Total Processed Files</div>
                <div class="stat-value">"#,
    );

    use std::fmt::Write;
    let _ = write!(&mut html, "{}", total_files);

    html.push_str(
        r#"</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 12px; color: var(--text-secondary);">Duplicate Clusters</div>
                <div class="stat-value">"#,
    );
    let _ = write!(&mut html, "{}", groups.len());

    html.push_str(
        r#"</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 12px; color: var(--text-secondary);">Reclaimable Space</div>
                <div class="stat-value">"#,
    );
    let _ = write!(&mut html, "{}", wasted_str);

    html.push_str(
        r#"</div>
            </div>
        </div>"#,
    );

    // Render Duplicate Groups
    if !groups.is_empty() {
        html.push_str("<h2>Duplicate Asset Clusters</h2>");
        for (idx, group) in groups.iter().enumerate() {
            let group_size = group.files.first().map(|f| f.size).unwrap_or(0);
            let _ = writeln!(
                &mut html,
                r#"<div class="group-card">
                    <div class="group-header">
                        <span>Group #{}: Hash [{}]</span>
                        <span>Master Size: {}</span>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>File Name</th>
                                <th>Format</th>
                                <th>Dimensions</th>
                                <th>Mipmaps</th>
                                <th>Size</th>
                                <th>Score</th>
                                <th>Full Path</th>
                            </tr>
                        </thead>
                        <tbody>"#,
                idx + 1,
                escape_html(&group.hash),
                crate::utils::helpers::format_size(group_size)
            );

            for (f_idx, file) in group.files.iter().enumerate() {
                let is_best = f_idx == 0;
                let name = Path::new(&file.path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy();
                let score_str = if is_best {
                    "<span class=\"best-badge\">BEST</span> 100.0%".to_string()
                } else {
                    format!("{:.1}%", file.similarity)
                };

                let _ = writeln!(
                    &mut html,
                    r#"<tr>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}x{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td class="path-cell">{}</td>
                    </tr>"#,
                    escape_html(&name),
                    escape_html(&file.compression_format),
                    file.width,
                    file.height,
                    file.mipmap_count,
                    crate::utils::helpers::format_size(file.size),
                    score_str,
                    escape_html(&file.path)
                );
            }

            html.push_str("</tbody></table></div>");
        }
    }

    // Render QC Issues
    if !qc_issues.is_empty() {
        html.push_str("<h2>Technical Quality Control Audit Issues</h2>");
        html.push_str(
            r#"<div class="group-card">
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>QC Issue Category</th>
                        <th>Details</th>
                        <th>Path</th>
                    </tr>
                </thead>
                <tbody>"#,
        );

        for issue in qc_issues {
            let name = Path::new(&issue.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let _ = writeln!(
                &mut html,
                r#"<tr>
                    <td>{}</td>
                    <td style="color: var(--highlight); font-weight: bold;">{}</td>
                    <td>{}</td>
                    <td class="path-cell">{}</td>
                </tr>"#,
                escape_html(&name),
                escape_html(&issue.issue),
                escape_html(&issue.details),
                escape_html(&issue.path)
            );
        }

        html.push_str("</tbody></table></div>");
    }

    // Render Inventory
    if !inventory.is_empty() {
        html.push_str("<h2>Asset Inventory Directory Audit</h2>");
        html.push_str(
            r#"<div class="group-card">
            <table>
                <thead>
                    <tr>
                        <th>File Name</th>
                        <th>Format</th>
                        <th>Dimensions</th>
                        <th>Mipmaps</th>
                        <th>Cubemap</th>
                        <th>Size</th>
                        <th>Path</th>
                    </tr>
                </thead>
                <tbody>"#,
        );

        for file in inventory {
            let name = Path::new(&file.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            let _ = writeln!(
                &mut html,
                r#"<tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}x{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td class="path-cell">{}</td>
                </tr>"#,
                escape_html(&name),
                escape_html(&file.compression_format),
                file.width,
                file.height,
                file.mipmap_count,
                if file.is_cubemap { "YES" } else { "NO" },
                crate::utils::helpers::format_size(file.size),
                escape_html(&file.path)
            );
        }

        html.push_str("</tbody></table></div>");
    }

    html.push_str("</div></body></html>");

    fs::write(out_path, html).context("Failed to write HTML report to file")?;
    Ok(())
}
