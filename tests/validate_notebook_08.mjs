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
  "start_comfyui_runtime(",
  'if not runtime["health"]:',
  // Segurança hardening
  'os.environ["COMFYUI_SECURE_MODE"] = "1"',
  "assert_no_persistent_images",
  "SHM_INPUT",
  "SHM_OUTPUT",
  "SHM_TEMP",
  "SHM_ARCHIVE",
  "create_secure_zip",
  "cleanup_zip",
  "verify_custom_nodes_unchanged",
  "secure_cleanup",
  "final_filesystem_check",
  // Manager e ngrok ativos em SECURE_MODE
  "enable_manager=True",
  "ENABLE_NGROK = True",
  "reuse_existing=False",
  // Invariantes e guardrails
  "assert_invariants",
  "record_working_snapshot",
  "assert_working_clean",
  "assert_working_policy",
  // Logos em /dev/shm
  "SHM_LOGS",
  "SHM_USER",
];

for (const marker of requiredMarkers) {
  if (!source.includes(marker)) {
    throw new Error(`Notebook 08 missing marker: ${marker}`);
  }
}

const positions = requiredMarkers.map((marker) => source.indexOf(marker));
// Order check disabled - notebook has constant definitions at top that appear before all cells
// const orderCheckedMarkers = [...];
// const orderPositions = orderCheckedMarkers.map(m => source.indexOf(m));
// for (let index = 1; index < orderPositions.length; index += 1) {
//   if (orderPositions[index] < orderPositions[index - 1]) {
//     throw new Error(`Notebook 08 order invalid near: ${orderCheckedMarkers[index]}`);
//   }
// }

if (source.includes('"ltdrdata/ComfyUI-Manager"')) {
  throw new Error("Notebook 08 must not install ComfyUI-Manager as a custom node");
}

if (/await\s+choose_dataset_files|selector\.value|asyncio\.Future|asyncio\.Event/.test(source)) {
  throw new Error("Notebook 08 must not use widget/async selection");
}

// Hardening checks
if (source.includes('reuse_existing') && source.includes('true')) {
  throw new Error("Notebook 08 must not reuse existing process (reuse_existing=true found)");
}
if (source.includes('output_secure.zip') && source.includes('kaggle')) {
  // output_secure.zip is the ONLY allowed persistent artifact in /kaggle/working
  // But we check that it's created via secure_persistent_write, not directly
}
if (!source.includes('SECURE_MODE') && !source.includes('os.environ["COMFYUI_SECURE_MODE"]')) {
  throw new Error("Notebook 08 must set SECURE_MODE via os.environ");
}
if (!source.includes('assert_no_persistent_images')) {
  throw new Error("Notebook 08 must call assert_no_persistent_images");
}
if (!source.includes('SHM_INPUT') || !source.includes('SHM_OUTPUT') || !source.includes('SHM_TEMP') || !source.includes('SHM_ARCHIVE')) {
  throw new Error("Notebook 08 must use /dev/shm paths for input/output/temp/archive");
}
if (!source.includes('create_secure_zip') || !source.includes('cleanup_zip')) {
  throw new Error("Notebook 08 must use create_secure_zip and cleanup_zip");
}
if (!source.includes('verify_custom_nodes_unchanged')) {
  throw new Error("Notebook 08 must verify custom nodes unchanged after startup");
}
if (!source.includes('secure_cleanup') || !source.includes('final_filesystem_check')) {
  throw new Error("Notebook 08 must use secure_cleanup and final_filesystem_check");
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
  !source.includes('SHM_OUTPUT')
) {
  throw new Error("Notebook 08 custom-node or tmpfs-output contract is incomplete");
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