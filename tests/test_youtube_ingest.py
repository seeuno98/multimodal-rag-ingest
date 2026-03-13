from __future__ import annotations

from src.ingest.youtube import _parse_vtt_to_segments


def test_parse_vtt_to_segments_handles_cue_settings(tmp_path) -> None:
    vtt_path = tmp_path / "sample.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:00:00.199 --> 00:00:02.510 align:start position:0%",
                "hello world",
                "",
                "00:02.510 --> 00:04.000 line:0%",
                "second segment",
                "",
            ]
        ),
        encoding="utf-8",
    )

    segments = _parse_vtt_to_segments(vtt_path)

    assert len(segments) == 2
    assert segments[0]["metadata"]["timestamp_start"] == 0.199
    assert segments[0]["metadata"]["timestamp_end"] == 2.51
    assert segments[1]["metadata"]["timestamp_start"] == 2.51
    assert segments[1]["metadata"]["timestamp_end"] == 4.0
