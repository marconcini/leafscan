# RAG corpus

`wikipedia.jsonl`: plain text of 28 English Wikipedia articles on the 26 PlantVillage
diseases, one JSON object per line (`title`, `url`, `revid`, `timestamp`, `license`, `classes`,
`text`). Snapshot taken on 2026-10-28 with `python -m leafscan.rag.corpus fetch`; versioned so
that experiments stay reproducible when the articles change.

## License and attribution

The text comes from Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Authors are listed in each
article's history; the exact revisions used are below. It was converted to plain text (the
TextExtracts API) and is otherwise unmodified. Any redistribution of this file must keep the
same license. This license applies to this folder only; the code is MIT.

| Article | Revision |
|---|---|
| [Apple scab](https://en.wikipedia.org/wiki/Apple_scab) | [1316621725](https://en.wikipedia.org/w/index.php?oldid=1316621725) |
| [Venturia inaequalis](https://en.wikipedia.org/wiki/Venturia_inaequalis) | [1380710615](https://en.wikipedia.org/w/index.php?oldid=1380710615) |
| [Botryosphaeria obtusa](https://en.wikipedia.org/wiki/Botryosphaeria_obtusa) | [1320582852](https://en.wikipedia.org/w/index.php?oldid=1320582852) |
| [Gymnosporangium juniperi-virginianae](https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae) | [1300966550](https://en.wikipedia.org/w/index.php?oldid=1300966550) |
| [Podosphaera clandestina](https://en.wikipedia.org/wiki/Podosphaera_clandestina) | [1324527363](https://en.wikipedia.org/w/index.php?oldid=1324527363) |
| [Erysiphaceae](https://en.wikipedia.org/wiki/Erysiphaceae) | [1371061050](https://en.wikipedia.org/w/index.php?oldid=1371061050) |
| [Podosphaera xanthii](https://en.wikipedia.org/wiki/Podosphaera_xanthii) | [1324528281](https://en.wikipedia.org/w/index.php?oldid=1324528281) |
| [Corn grey leaf spot](https://en.wikipedia.org/wiki/Corn_grey_leaf_spot) | [1366675422](https://en.wikipedia.org/w/index.php?oldid=1366675422) |
| [Puccinia sorghi](https://en.wikipedia.org/wiki/Puccinia_sorghi) | [1380701438](https://en.wikipedia.org/w/index.php?oldid=1380701438) |
| [Northern corn leaf blight](https://en.wikipedia.org/wiki/Northern_corn_leaf_blight) | [1313233555](https://en.wikipedia.org/w/index.php?oldid=1313233555) |
| [Setosphaeria turcica](https://en.wikipedia.org/wiki/Setosphaeria_turcica) | [1296531162](https://en.wikipedia.org/w/index.php?oldid=1296531162) |
| [Black rot (grape disease)](https://en.wikipedia.org/wiki/Black_rot_(grape_disease)) | [1381547052](https://en.wikipedia.org/w/index.php?oldid=1381547052) |
| [Esca (grape disease)](https://en.wikipedia.org/wiki/Esca_(grape_disease)) | [1184184781](https://en.wikipedia.org/w/index.php?oldid=1184184781) |
| [Grapevine trunk disease](https://en.wikipedia.org/wiki/Grapevine_trunk_disease) | [1332164226](https://en.wikipedia.org/w/index.php?oldid=1332164226) |
| [Pseudocercospora vitis](https://en.wikipedia.org/wiki/Pseudocercospora_vitis) | [1321714350](https://en.wikipedia.org/w/index.php?oldid=1321714350) |
| [Citrus greening disease](https://en.wikipedia.org/wiki/Citrus_greening_disease) | [1378499377](https://en.wikipedia.org/w/index.php?oldid=1378499377) |
| [Xanthomonas arboricola pv. pruni](https://en.wikipedia.org/wiki/Xanthomonas_arboricola_pv._pruni) | [1372084729](https://en.wikipedia.org/w/index.php?oldid=1372084729) |
| [Bacterial leaf spot of peppers and tomato](https://en.wikipedia.org/wiki/Bacterial_leaf_spot_of_peppers_and_tomato) | [1380779742](https://en.wikipedia.org/w/index.php?oldid=1380779742) |
| [Alternaria solani](https://en.wikipedia.org/wiki/Alternaria_solani) | [1380651171](https://en.wikipedia.org/w/index.php?oldid=1380651171) |
| [Phytophthora infestans](https://en.wikipedia.org/wiki/Phytophthora_infestans) | [1380700327](https://en.wikipedia.org/w/index.php?oldid=1380700327) |
| [Diplocarpon earlianum](https://en.wikipedia.org/wiki/Diplocarpon_earlianum) | [1225303728](https://en.wikipedia.org/w/index.php?oldid=1225303728) |
| [Tomato leaf mold](https://en.wikipedia.org/wiki/Tomato_leaf_mold) | [1321692430](https://en.wikipedia.org/w/index.php?oldid=1321692430) |
| [Passalora fulva](https://en.wikipedia.org/wiki/Passalora_fulva) | [1378623130](https://en.wikipedia.org/w/index.php?oldid=1378623130) |
| [Septoria lycopersici](https://en.wikipedia.org/wiki/Septoria_lycopersici) | [1344451937](https://en.wikipedia.org/w/index.php?oldid=1344451937) |
| [Tetranychus urticae](https://en.wikipedia.org/wiki/Tetranychus_urticae) | [1355026502](https://en.wikipedia.org/w/index.php?oldid=1355026502) |
| [Corynespora cassiicola](https://en.wikipedia.org/wiki/Corynespora_cassiicola) | [1380660673](https://en.wikipedia.org/w/index.php?oldid=1380660673) |
| [Tomato yellow leaf curl virus](https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus) | [1374302172](https://en.wikipedia.org/w/index.php?oldid=1374302172) |
| [Tomato mosaic virus](https://en.wikipedia.org/wiki/Tomato_mosaic_virus) | [1334341165](https://en.wikipedia.org/w/index.php?oldid=1334341165) |
