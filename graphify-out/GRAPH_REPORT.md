# Graph Report - ElectionAppFinalMix  (2026-06-19)

## Corpus Check
- 9 files · ~34,194 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 101 nodes · 126 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 93% EXTRACTED · 6% INFERRED · 1% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d8711012`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Flask Routes & Templates|Flask Routes & Templates]]
- [[_COMMUNITY_Admin & Auth Routes|Admin & Auth Routes]]
- [[_COMMUNITY_Model Training & Logging|Model Training & Logging]]
- [[_COMMUNITY_Image Processing Pipeline|Image Processing Pipeline]]
- [[_COMMUNITY_Face Detection (YOLOHaar)|Face Detection (YOLO/Haar)]]
- [[_COMMUNITY_Face Recognition Model|Face Recognition Model]]
- [[_COMMUNITY_Label Map Versions|Label Map Versions]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_Admin Password|Admin Password]]
- [[_COMMUNITY_Graphify Dependency|Graphify Dependency]]
- [[_COMMUNITY_Gemini Config|Gemini Config]]

## God Nodes (most connected - your core abstractions)
1. `Flask app instance` - 22 edges
2. `retrain_face_model_from_existing_images()` - 12 edges
3. `write_train_log()` - 7 edges
4. `detect_faces()` - 7 edges
5. `ConsoleTrainingLogger` - 7 edges
6. `register_post()` - 7 edges
7. `normalize_all_existing_images()` - 6 edges
8. `normalize_image_for_camera_variation()` - 6 edges
9. `load_face_model()` - 6 edges
10. `detect_and_crop_face()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Flask app instance` --references--> `graphifyy`  [AMBIGUOUS]
  app.py → requirements.txt
- `best model/label_map.json` --semantically_similar_to--> `label_map.json`  [INFERRED] [semantically similar]
  Face_Recog_App/model/best model/label_map.json → app.py
- `best model/best model/label_map.json` --semantically_similar_to--> `label_map.json`  [INFERRED] [semantically similar]
  Face_Recog_App/model/best model/best model/label_map.json → app.py
- `best model/best model/best model/label_map.json` --semantically_similar_to--> `label_map.json`  [INFERRED] [semantically similar]
  Face_Recog_App/model/best model/best model/best model/label_map.json → app.py
- `model/label_map.json` --semantically_similar_to--> `label_map.json`  [INFERRED] [semantically similar]
  Face_Recog_App/model/label_map.json → app.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Model retraining pipeline** — app_retrain_face_model_from_existing_images, app_load_data_rgb, cnn_architecture, app_face_cnn_model_keras, app_label_map_json [EXTRACTED 1.00]
- **Face verification flow** — app_flask_app, app_load_face_model, app_detect_and_crop_face, app_detect_faces, yolo_haar_fallback, camera_normalization [EXTRACTED 1.00]
- **Voting data flow** — app_flask_app, app_users_csv, app_votes_csv, vote_html, result_html [EXTRACTED 1.00]

## Communities (12 total, 1 thin omitted)

### Community 0 - "Flask Routes & Templates"
Cohesion: 0.09
Nodes (17): Hardcoded admin password rationale, ADMIN_PASSWORD, Flask app instance, users.csv, votes.csv, Graphify dependency rationale, flask, graphifyy (+9 more)

### Community 1 - "Admin & Auth Routes"
Cohesion: 0.11
Nodes (10): cv2_imread_utf8(), cv2_imwrite_utf8(), load_data_rgb(), normalize_all_existing_images(), normalize_image_for_camera_variation(), normalize_images(), Normalize รูปภาพเก่าทั้งหมดที่มีอยู่แล้ว, Normalize ภาพเพื่อลดผลกระทบจากกล้องต่างกัน (+2 more)

### Community 2 - "Model Training & Logging"
Cohesion: 0.22
Nodes (10): ConsoleTrainingLogger, แสดงสถานะการเทรนใน console แบบอ่านง่าย, เทรนโมเดลใหม่จากรูปผู้ใช้ที่มีอยู่ เพื่อซ่อมกรณีไฟล์โมเดลเก่าโหลดไม่ได้, Route สำหรับสั่งเทรนโมเดลใหม่ด้วยมือ (ต้องเป็น admin), Fallback retrain แบบ form submit (ไม่พึ่ง JavaScript), retrain_face_model_from_existing_images(), retrain_model(), retrain_model_sync() (+2 more)

### Community 3 - "Image Processing Pipeline"
Cohesion: 0.29
Nodes (5): Commands, Development and Maintenance, High-Level Code Architecture and Structure, Running the Application, Setup and Installation

### Community 4 - "Face Detection (YOLO/Haar)"
Cohesion: 0.28
Nodes (9): detect_and_crop_face(), detect_faces(), face_cascade (Haar), index(), process_upload_to_cv2(), ตรวจจับใบหน้าโดยใช้ YOLO ก่อน (ถ้ามี), แล้ว fallback ไป Haar Cascade     คืนค่า, register_post(), yolo_model (+1 more)

### Community 5 - "Face Recognition Model"
Cohesion: 0.21
Nodes (7): CompatibleInputLayer, face_cnn_model.keras, load_face_model(), load_face_model_with_patched_keras_config(), รองรับโมเดลเก่าที่ serialize ด้วย key ชื่อ batch_shape, โหลดโมเดลแบบทนทานกับไฟล์ .keras เก่า/ใหม่, _replace_batch_shape_key()

### Community 6 - "Label Map Versions"
Cohesion: 0.40
Nodes (5): label_map.json, best model/label_map.json, best model/best model/label_map.json, best model/best model/best model/label_map.json, model/label_map.json

### Community 7 - "Project Documentation"
Cohesion: 0.67
Nodes (3): CLAUDE.md project instructions, Modification History, Skill Reference

### Community 8 - "Admin Password"
Cohesion: 0.50
Nodes (3): 2026‑06‑18, 2026‑06‑18, ElectionAppFinalMix – Modification History

### Community 9 - "Graphify Dependency"
Cohesion: 0.50
Nodes (3): ElectionAppFinalMix – Skill Reference, Main Components, Overview

## Ambiguous Edges - Review These
- `Flask app instance` → `graphifyy`  [AMBIGUOUS]
  requirements.txt · relation: references

## Knowledge Gaps
- **32 isolated node(s):** `High-Level Code Architecture and Structure`, `Setup and Installation`, `Running the Application`, `Development and Maintenance`, `Overview` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Flask app instance` and `graphifyy`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `Flask app instance` connect `Flask Routes & Templates` to `Model Training & Logging`, `Face Recognition Model`?**
  _High betweenness centrality (0.305) - this node is a cross-community bridge._
- **Why does `retrain_face_model_from_existing_images()` connect `Model Training & Logging` to `Flask Routes & Templates`, `Admin & Auth Routes`, `Face Detection (YOLO/Haar)`, `Face Recognition Model`, `Label Map Versions`?**
  _High betweenness centrality (0.253) - this node is a cross-community bridge._
- **Why does `load_face_model()` connect `Face Recognition Model` to `Flask Routes & Templates`, `Admin & Auth Routes`, `Face Detection (YOLO/Haar)`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **What connects `Normalize รูปภาพเก่าทั้งหมดที่มีอยู่แล้ว`, `Normalize ภาพเพื่อลดผลกระทบจากกล้องต่างกัน`, `ตรวจจับใบหน้าโดยใช้ YOLO ก่อน (ถ้ามี), แล้ว fallback ไป Haar Cascade     คืนค่า` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Flask Routes & Templates` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._
- **Should `Admin & Auth Routes` be split into smaller, more focused modules?**
  _Cohesion score 0.11067193675889328 - nodes in this community are weakly interconnected._