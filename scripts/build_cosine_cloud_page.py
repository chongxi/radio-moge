"""Build a standalone point-cloud page from a completed fixed20 export."""

import argparse
import base64
import gzip
import html
import json
from pathlib import Path
import struct
import subprocess


def build(root, output):
    repo = Path(__file__).resolve().parents[1]
    report = json.loads((root / "data.json").read_text())
    assert report["status"] == "complete" and isinstance(report["checkpoint_step"], int)
    assert report["names"] in (["student", "teacher"], ["student", "parent", "teacher"])
    count = len(report["names"])
    assert len(report["labels"]) == count
    assert len(report["samples"]) == len(set(report["selected_ids"])) == 20
    assert [s["id"] for s in report["samples"]] == report["selected_ids"]
    for sample in report["samples"]:
        n = ((sample["width"] + sample["stride"] - 1) // sample["stride"]) * (
            (sample["height"] + sample["stride"] - 1) // sample["stride"]
        )
        blob = gzip.decompress(base64.b64decode(sample["payload"]))
        assert len(blob) == n * 5 * count
        for branch in range(count):
            mask = blob[4 * count * n + branch * n:4 * count * n + (branch + 1) * n]
            assert sum(mask) > 1000 and set(mask) <= {0, 1}
            depths = struct.unpack_from(f"<{n}f", blob, branch * n * 4)
            assert all(0 < d < float("inf") for d, valid in zip(depths, mask) if valid)
        assert sample["projection_error_m"] < 1e-4
    bundle = root / "viewer.bundle.js"
    subprocess.run(["bun", "build", str(repo / "web/cosine-clouds.js"), "--target=browser",
                    "--minify", "--outfile=" + str(bundle)], check=True)
    display = {k: report[k] for k in ["labels", "names", "samples", "summary", "checkpoint_step",
                                    "selection_metric", "selection_value"]}
    page = (repo / "web/cosine-clouds.html").read_text()
    start = page.index('<div id="panels">')
    end = page.index('<p id="status"', start)
    colors = ['#ffae72', '#83b5ff', '#44d5b1'] if count == 3 else ['#ffae72', '#44d5b1']
    cards = []
    for i, label in enumerate(report['labels']):
        cards.append(f'<section class="card" id="card{i}"><h2><i class="dot" style="background:{colors[i]}"></i>'
                     f'<span id="title{i}">{html.escape(label)}</span></h2><div class="cloud" id="view{i}"></div>'
                     f'<div class="foot"><span id="count{i}"></span><button id="download{i}">下载 PLY</button></div></section>')
    page = page[:start] + '<div id="panels">' + ''.join(cards) + '</div>\n  ' + page[end:]
    if count == 3:
        page = page.replace('</style>', '#panels:not(.overlay){grid-template-columns:repeat(3,minmax(0,1fr))}'
                            '#summary{grid-template-columns:repeat(3,minmax(0,1fr))}'
                            '@media(max-width:900px){#panels:not(.overlay){grid-template-columns:1fr}}'
                            '@media(max-width:450px){#summary{grid-template-columns:1fr}}\n</style>')
    page = page.replace('<strong>0.959922</strong>', f'<strong>{report["selection_value"]:.6f}</strong>')
    checkpoint_label = report.get('view', {}).get('checkpoint_label',
                                                f'第 {report["checkpoint_step"]:,} 步 · best-spatial.pt')
    page = page.replace('第 91,367 步 · best-spatial.pt', html.escape(checkpoint_label))
    if report.get("view"):
        view = report["view"]
        page = page.replace("Cosine 最佳权重 × MoGe-2", html.escape(view["title"]))
        page = page.replace("同一输入图像 · 同步三维视角 · 各自预测的几何与尺度", html.escape(view["subtitle"]))
        page = page.replace("最佳验证 spatial cosine", html.escape(view["badge_label"]))
        old_protocol = page[page.index("<p>两侧使用相同 RGB、"):page.index("<p>展示点云")]
        page = page.replace(old_protocol, "<p>" + html.escape(view["protocol"]) + "</p>")
        page = page.replace("以上为这 20 张图的平均值；下表为当前图像。AbsRel 和 RMSE 越低越好。",
                            html.escape(view["caveat"]) + " 下表为当前图像；AbsRel 和 RMSE 越低越好。")
        if view.get('weights'):
            start = page.index('<p>左侧使用 C-RADIOv4-H')
            end = page.index('</p>', start) + 4
            page = page[:start] + '<p>' + html.escape(view['weights']) + '</p>' + page[end:]
    page = page.replace("__DATA__", json.dumps(display, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__BUNDLE__", bundle.read_text().replace("</script", "<\\/script"))
    for destination in [root / "index.html", output]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".html.tmp")
        temporary.write_text(page)
        temporary.replace(destination)
    print(json.dumps(dict(stage="built", scenes=20, bytes=output.stat().st_size, output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.root, args.output)
