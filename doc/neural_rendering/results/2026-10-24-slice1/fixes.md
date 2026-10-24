# [result] Slice 1 review fixes pushed and merged (neural-p2 @ e2ffb2ff33d)

Plain version: the crash on a damaged `.nnm` is fixed, slice 1 is merged into the coordination branch (merge commit 036e976fd47, no rebase), and slice 2 now sits on top of it.

| review item | fix |
|---|---|
| Wrapping block bounds | begin > size, size > file_size - begin and end < begin are all checked before use; new `SceneNeural` case with `latent_offset = 0xFFFFFFFFFFFFFFC0` is rejected with "past the end" |
| 32-bit offsets | loader rejects a latent block over 2^31 values; `device_update` logs an error and uploads empty models if a total exceeds 2^32 |
| MD5 truncation | hashed in 1 GiB chunks |

Gates: build exit 0; `ctest -R "^cycles$"` 228/228 including the new case. I did not re-run the node gate or `^cycles_` after this small loader change.

Slice 2 (`neural-p2-closure`, now includes this fix) is ready for review: see the earlier result (comment 6382224340). Slice 2 already has the `UV` input; the `Tangent` input and anisotropy are not done. Questions still open from that comment: merge slice 2 after review, and next step (UV-varying material vs fit accuracy). My default if I hear nothing: a UV-varying test material (textured Principled) to exercise the latent grid.
