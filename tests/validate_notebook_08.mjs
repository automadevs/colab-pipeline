import fs from "node:fs";

const notebookPath = "kaggle_runtime/08_master_pipeline.ipynb";
const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
const source = notebook.cells
  .filter((cell) => cell.metadata?.language === "python")
  .flatMap((cell) => cell.source || [])
  .join("");

const requiredMarkers = [
  'git", "clone"',
  'shutil.copytree(REPO_DIR / "scripts", SCRIPTS_DIR)',
  "from gpu_detect import detect_gpu",
  "test_drive_connection",
  "CUSTOM_NODES = [",
  "setup_comfyui(",
  "SELECTED_FILES = select_dataset_files(",
  "sync_dataset_to_local(",
  "selected_files=SELECTED_FILES",
  "start_comfyui_runtime(",
  'if not runtime["health"]:',
  "runtime['public_url']",
];

for (const marker of requiredMarkers) {
  if (!source.includes(marker)) {
    throw new Error(`Notebook 08 missing marker: ${marker}`);
  }
}

const positions = requiredMarkers.map((marker) => source.indexOf(marker));
for (let index = 1; index < positions.length; index += 1) {
  if (positions[index] < positions[index - 1]) {
    throw new Error(`Notebook 08 order invalid near: ${requiredMarkers[index]}`);
  }
}

if (source.includes('"ltdrdata/ComfyUI-Manager"')) {
  throw new Error("Notebook 08 must not install ComfyUI-Manager as a custom node");
}

if (/await\s+choose_dataset_files|selector\.value|asyncio\.Future|asyncio\.Event/.test(source)) {
  throw new Error("Notebook 08 must not use widget/async selection");
}

const syncSource = fs.readFileSync("scripts/kaggle_sync.py", "utf8");
for (const marker of [
  "def parse_model_selection(",
  "def select_dataset_files(",
  "input_fn=input",
  "parse_model_selection(input_fn(\"\\n> Seleção: \"), candidate_paths)",
]) {
  if (!syncSource.includes(marker)) {
    throw new Error(`kaggle_sync.py missing synchronous selection marker: ${marker}`);
  }
}

if (
  !source.includes('"cubiq/ComfyUI_essentials"') ||
  !source.includes('"lbouaraba/comfyui-krea2edit"') ||
  !source.includes('OUTPUT_DIR = COMFYUI_DIR / "output"')
) {
  throw new Error("Notebook 08 custom-node or local-output contract is incomplete");
}

const setupSource = fs.readFileSync("scripts/comfyui_setup.py", "utf8");
if (setupSource.indexOf("health_check(") > setupSource.indexOf("start_ngrok_tunnel")) {
  throw new Error("ComfyUI health check must precede ngrok startup");
}
if (!setupSource.includes('"--output-directory"') || !setupSource.includes("/output")) {
  throw new Error("ComfyUI output directory contract is incomplete");
}
if (setupSource.includes('"--listen",\n        "0.0.0.0"')) {
  throw new Error("ComfyUI must not bind publicly by default");
}

console.log("Notebook 08 structural flow OK");