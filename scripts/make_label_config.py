#!/usr/bin/env python3
"""Emit the Label Studio labelling config from schema/rating_schema.json.

Single source of truth: edit the schema, re-run this, push the result to the
project. Keeps the rater-facing wording and the scored schema from drifting.

    python scripts/make_label_config.py > annotation/label_config.xml
"""
import json
import pathlib
import sys
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schema" / "rating_schema.json"

BLOCK = """  <View style="margin:14px 0; padding:12px; border:1px solid #ddd; border-radius:4px;">
    <Header value="{title}" size="4"/>
    <Header value="{legend}" size="6" style="color:#666; font-weight:400; font-size:12px; margin:2px 0 8px;"/>
    <Choices name="{id}" toName="vig" choice="single" required="true" showInline="true">
{choices}
    </Choices>
  </View>"""


def build(schema: dict) -> str:
    parts = []
    for d in schema["dimensions"]:
        a = d["anchors"]
        legend = f"-3 = {a['-3']}   |   0 = {a['0']}   |   +3 = {a['3']}"
        if "note" in d:
            legend += f"   |   {d['note']}"
        choices = "\n".join(
            f'      <Choice value="{v}" alias="{v}"/>' for v in range(-3, 4)
        )
        parts.append(
            BLOCK.format(
                id=d["id"],
                title=escape(f"{d['id']}: {d['prompt']}"),
                legend=escape(legend),
                choices=choices,
            )
        )
    return (
        '<View>\n'
        '  <View style="margin-bottom:16px; padding:14px; background:#f7f7f7; '
        'font-size:17px; line-height:1.5; border-radius:4px;">\n'
        '    <Text name="vig" value="$text"/>\n'
        '    <Text name="vid" value="$vignette_id" style="color:#888; font-size:11px;"/>\n'
        '  </View>\n'
        + "\n".join(parts)
        + "\n</View>\n"
    )


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    missing = [d["id"] for d in schema["dimensions"] if "anchors" not in d]
    if missing:
        print(f"schema dimensions missing anchors: {missing}", file=sys.stderr)
        return 1
    sys.stdout.write(build(schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
