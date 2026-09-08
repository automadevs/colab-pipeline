import fs from "node:fs";

const notebooks = [
  "kaggle_runtime/05_download_to_local.ipynb",
  "kaggle_runtime/07_sync_robust.ipynb",
  "kaggle_runtime/08_master_pipeline.ipynb",
];

for (const notebookPath of notebooks) {
  const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
  const source = notebook.cells
    .filter((cell) => cell.metadata?.language === "python")
    .flatMap((cell) => cell.source || [])
    .join("");

  for (const marker of [
    "select_dataset_files(",
    "SELECTED_FILES",
    "sync_dataset_to_local(",
    "selected_files=SELECTED_FILES",
  ]) {
    if (!source.includes(marker)) {
      throw new Error(`${notebookPath} missing marker: ${marker}`);
    }
  }

  if (/await\s+choose_dataset_files|selector\.value|asyncio\.Future|asyncio\.Event|ipywidgets/.test(source)) {
    throw new Error(`${notebookPath} still contains asynchronous/widget selection`);
  }
}

console.log("Synchronous selection flow OK in all Kaggle runtime notebooks");
