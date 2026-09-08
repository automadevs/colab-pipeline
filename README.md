# Pipeline Civitai → Kaggle Dataset → Kaggle Notebook (ComfyUI)

Arquitetura definitiva para transferência e execução de modelos ComfyUI.

---

## Arquitetura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│    CIVITAI      │     │  GOOGLE COLAB    │     │   KAGGLE DATASET    │
│   (Modelos)     │────▶│  (Transferência) │────▶│  (Modelos Core)     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   GOOGLE DRIVE  │◀───▶│  KAGGLE NOTEBOOK │◀───▶│   KAGGLE DATASET    │
│  (Outputs/Logs) │     │   (Compute/GPU)  │     │  (Modelos Core)     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                               │
                               ▼
                         ┌───────────────┐
                         │   COMFYUI     │
                         │  (SSD Local)  │
                         └───────────────┘
```

**Separação de responsabilidades:**
- **Civitai** → Origem dos modelos
- **Google Colab** → Estação de transferência (Civitai → Kaggle Dataset)
- **Kaggle Dataset** → Armazenamento persistente de modelos core (`automamermaid/comfydocs`) — checkpoints, LoRAs, VAEs, text_encoders, controlnet, etc.
- **GitHub** → Código (ComfyUI, custom nodes, scripts deste repo)
- **Google Drive** → Outputs (imagens geradas), logs, workflows salvos, metadata — arquivos leves, muitos, não vão pro GitHub
- **Kaggle Notebook** → Compute/GPU, executa ComfyUI, sincroniza modelos do Dataset para SSD local, envia outputs para Drive

**Separação de responsabilidades:**
- **Civitai** → Origem dos modelos
- **Google Colab** → Estação de transferência (Civitai → Kaggle Dataset)
- **Kaggle Dataset** → Armazenamento persistente de modelos (`automamermaid/comfydocs`)
- **GitHub** → Código (ComfyUI, custom nodes, scripts)
- **Kaggle Notebook** → Compute/GPU, executa ComfyUI, sincroniza modelos do Dataset para SSD local

---

## Estrutura do Projeto

```
colab_pipeline/
├── colab_transfer/           # Notebooks para rodar no GOOGLE COLAB
│   ├── 01_inspecao.ipynb     # Verifica auth, dataset, staging
│   ├── 02_download.ipynb     # Civitai → /content/kaggle_staging
│   ├── 03_upload_kaggle.ipynb # Staging → Kaggle Dataset (CLI)
│   └── 04_upload_kagglehub.ipynb # Fallback upload via kagglehub
│
├── kaggle_runtime/           # Notebooks para rodar no KAGGLE NOTEBOOK
│   ├── 05_download_to_local.ipynb  # Dataset → /kaggle/working/ComfyUI/models/
│   ├── 06_comfyui_setup.ipynb      # GitHub → ComfyUI local + custom nodes
│   ├── 07_sync_robust.ipynb        # Sync modelos Dataset → SSD (idempotente)
│   ├── 08_master_pipeline.ipynb    # Orquestrador completo (inicialização total)
│   └── 09_sync_outputs.ipynb       # Sync outputs/workflows ↔ Google Drive
│
├── scripts/                  # Scripts Python compartilhados
│   ├── civitai_download.py   # Download robusto da Civitai
│   ├── kaggle_upload.py      # Upload para Kaggle Dataset (CLI + kagglehub)
│   ├── kaggle_sync.py        # Sync Dataset → SSD local (idempotente, seletivo)
│   ├── kaggle_drive_sync.py  # Sync outputs/workflows ↔ Google Drive
│   ├── comfyui_setup.py      # Instala/atualiza ComfyUI + custom nodes
│   └── gpu_detect.py         # Detecção de GPU
│
└── README.md                 # Esta documentação
```

---

## Fluxo 1: Adicionar Modelo (Colab → Kaggle Dataset)

**Executar no Google Colab:**

```bash
# 1. Configure Secrets no Colab (⚙️ → Secrets):
#    CIVITAI_TOKEN = seu token da Civitai
#    KAGGLE_USERNAME + KAGGLE_KEY = credenciais Kaggle

# 2. Execute em ordem:
#    colab_transfer/01_inspecao.ipynb      # Verifica ambiente
#    colab_transfer/02_download.ipynb      # Baixa da Civitai (~12GB)
#    colab_transfer/03_upload_kaggle.ipynb # Sobe para Kaggle Dataset
#    # Se der 403: colab_transfer/04_upload_kagglehub.ipynb
```

**O PC do usuário NÃO participa da transferência.** Tudo roda na nuvem.

---

## Fluxo 2: Iniciar ComfyUI (Kaggle Notebook)

**Executar no Kaggle Notebook (com GPU):**

```bash
# Opção A: Pipeline completo (recomendado na primeira vez)
kaggle_runtime/08_master_pipeline.ipynb

# Opção B: Passo a passo
kaggle_runtime/05_download_to_local.ipynb  # Sync modelos
kaggle_runtime/06_comfyui_setup.ipynb      # Instala ComfyUI + custom nodes
# Depois inicie manualmente:
cd /kaggle/working/ComfyUI && python main.py --listen 0.0.0.0 --port 8188

# Opção C: Sync diário (operação cotidiana)
kaggle_runtime/07_sync_robust.ipynb
```

---

## Scripts Compartilhados (CLI)

Todos os notebooks usam scripts em `scripts/` para lógica reutilizável:

```bash
# Download Civitai
python scripts/civitai_download.py \
  --model-name "lustifyNSFWCheckpoint_v10Krea2.safetensors" \
  --staging-dir /content/kaggle_staging

# Upload Kaggle Dataset
python scripts/kaggle_upload.py \
  --model-name "lustifyNSFWCheckpoint_v10Krea2.safetensors" \
  --dataset "automamermaid/comfydocs" \
  --method cli  # ou kagglehub

# Sync Kaggle → Local (Kaggle Notebook)
python scripts/kaggle_sync.py \
  --dataset "automamermaid/comfydocs" \
  --target-dir /kaggle/working/ComfyUI/models \
  --categories checkpoints loras vae

# Setup ComfyUI (Kaggle Notebook)
python scripts/comfyui_setup.py \
  --comfyui-dir /kaggle/working/ComfyUI \
  --custom-nodes "ltdrdata/ComfyUI-Manager" "cubiq/ComfyUI_essentials" \
  --start --health-check

# Detectar GPU
python scripts/gpu_detect.py --recommend

# Sync Outputs ↔ Google Drive (Kaggle Notebook)
python scripts/kaggle_drive_sync.py \
  --action push \
  --drive-base "Automa/ComfyUI" \
  --env kaggle

# Pull workflows do Drive
python scripts/kaggle_drive_sync.py \
  --action pull \
  --drive-base "Automa/ComfyUI" \
  --env kaggle
```

---

## Configuração Inicial

### Google Colab (Transferência)

1. Abra qualquer notebook em `colab_transfer/`
2. Configure **Secrets** (ícone de chave ⚙️):
   - `CIVITAI_TOKEN`: Token da Civitai (Settings → API Keys)
   - `KAGGLE_USERNAME`: Seu username Kaggle
   - `KAGGLE_KEY`: Sua API key Kaggle (Account → Create New Token)
3. Execute as células

### Kaggle Notebook (Runtime)

1. Crie novo Notebook no Kaggle
2. Ative **GPU** (Settings → Accelerator → GPU)
3. Configure **Secrets** (ícone de chave):
   - `KAGGLE_USERNAME` + `KAGGLE_KEY` (mesmo do Colab)
   - `GDRIVE_SERVICE_ACCOUNT_JSON`: Service Account JSON com acesso ao Google Drive (para sync de outputs)
4. Faça upload dos arquivos:
   - Arraste pasta `scripts/` para `/kaggle/working/scripts/`
   - Arraste notebooks de `kaggle_runtime/` para o notebook
5. Execute `08_master_pipeline.ipynb`

### Google Drive (Outputs & Logs)

**Estrutura no Drive:**
```
/Meu Drive/Automa/ComfyUI/
├── outputs/          # Imagens geradas (organizadas por data/sessão)
├── workflows/        # Workflows salvos (.json)
├── logs/             # Logs do ComfyUI
└── metadata/         # CSVs/JSONs com metadados de geração
```

**Configuração do Service Account (Kaggle):**
1. No Google Cloud Console: crie Service Account → Role: Editor → Create Key (JSON)
2. Compartilhe a pasta `Automa/ComfyUI` no Drive com o email do Service Account (Editor)
3. No Kaggle: Secrets → `GDRIVE_SERVICE_ACCOUNT_JSON` = conteúdo do JSON

**No Colab:** Usa `google.colab.drive.mount()` nativo (sem service account).

**Sync manual:** Execute `kaggle_runtime/09_sync_outputs.ipynb` para push/pull.

---

## Modelo Atual

| Campo | Valor |
|-------|-------|
| **Dataset** | `automamermaid/comfydocs` |
| **Modelo** | `lustifyNSFWCheckpoint_v10Krea2.safetensors` |
| **Civitai Version ID** | `3112728` |
| **Civitai File ID** | `2997637` |
| **Formato** | FP8 |
| **Tamanho** | ~11.94 GB |
| **Download URL** | `https://civitai.com/api/download/models/3112728?fileId=2997637` |

**Nome ANTIGO (não usar):** `lustify-v10-krea-turbo-fp8.safetensors`

---

## Categorias de Modelos Suportadas

O Dataset `automamermaid/comfydocs` deve armazenar:

```
checkpoints/           # Checkpoints principais (.safetensors, .ckpt)
diffusion_models/      # Modelos de difusão (Flux, SD3, etc)
loras/                 # LoRAs (.safetensors)
vae/                   # VAEs
text_encoders/         # Text encoders (T5, CLIP, etc)
clip/                  # CLIP models
controlnet/            # ControlNet models
upscale_models/        # Upscalers (ESRGAN, 4xUltrasharp, etc)
video_models/          # Modelos de vídeo (SVD, AnimateDiff, etc)
embeddings/            # Textual inversions / embeddings
```

---

## Comandos Rápidos

### Sync Diário (Kaggle Notebook)
```bash
# No terminal do Kaggle Notebook ou via notebook 07_sync_robust.ipynb
python /kaggle/working/scripts/kaggle_sync.py \
  --dataset automamermaid/comfydocs \
  --target-dir /kaggle/working/ComfyUI/models \
  --categories checkpoints loras vae controlnet
```

### Adicionar Novo Modelo (Colab)
```bash
# 1. Baixar para staging
python scripts/civitai_download.py \
  --model-name "novo_modelo.safetensors" \
  --model-version-id "XXXXXX" \
  --file-id "YYYYYY" \
  --staging-dir /content/kaggle_staging

# 2. Upload para Dataset
python scripts/kaggle_upload.py \
  --model-name "novo_modelo.safetensors" \
  --dataset "automamermaid/comfydocs" \
  --version-notes "Add novo_modelo"
```

### Iniciar ComfyUI Manualmente
```bash
cd /kaggle/working/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header
```

---

## Troubleshooting

| Problema | Causa | Solução |
|----------|-------|---------|
| Kaggle 403 no upload | Sem permissão de edição | Verifique se é owner/colaborador do dataset `automamermaid/comfydocs` |
| Download 0 bytes | Token Civitai inválido | Verifique `CIVITAI_TOKEN` nos Secrets |
| Modelo não encontrado no dataset | Upload anterior falhou | Re-execute upload (03 ou 04) |
| Tamanho divergente | Download parcial | Delete staging e rebaixe |
| kagglehub sem dataset_upload | Versão antiga | `pip install -U kagglehub` |
| ComfyUI não inicia | Dependências faltando | Execute `06_comfyui_setup.ipynb` novamente |
| VRAM insuficiente | Modelo muito grande | Use `--lowvram` ou `--cpu` no ComfyUI |
| Drive não monta (Kaggle) | Service Account inválido | Verifique `GDRIVE_SERVICE_ACCOUNT_JSON` nos Secrets |
| Drive permission denied | Pasta não compartilhada | Compartilhe `Automa/ComfyUI` com email do Service Account (Editor) |
| rclone não encontrado | Não instalado | `pip install rclone` ou instale via apt |
| Outputs não aparecem no Drive | Output dir incorreto | Verifique `--output-directory` no ComfyUI |

---

## Checklist de Validação

### Colab Transfer
- [ ] Python 3.10+ no Colab
- [ ] Kaggle CLI ≥ 1.5.12
- [ ] `~/.kaggle/kaggle.json` configurado
- [ ] `CIVITAI_TOKEN` nos Secrets
- [ ] Dataset `automamermaid/comfydocs` acessível
- [ ] Permissão de edição no dataset
- [ ] Modelo `lustifyNSFWCheckpoint_v10Krea2.safetensors` (~11.94 GB)
- [ ] Upload cria nova versão no Kaggle (sem erro 403)

### Kaggle Runtime
- [ ] GPU ativada (T4, P100, ou A100)
- [ ] `kaggle.json` nos Secrets
- [ ] `GDRIVE_SERVICE_ACCOUNT_JSON` nos Secrets
- [ ] Scripts em `/kaggle/working/scripts/`
- [ ] ComfyUI clonado do GitHub
- [ ] Custom nodes instalados
- [ ] Modelos sincronizados do Dataset → SSD
- [ ] Google Drive montado (`/kaggle/working/gdrive/Automa/ComfyUI`)
- [ ] ComfyUI inicia com `--output-directory` no Drive
- [ ] ComfyUI responde em `:8188`
- [ ] Modelo principal aparece no "Load Checkpoint"
- [ ] Imagem gerada aparece no Drive (`Automa/ComfyUI/outputs/`)

---

## Próximos Passos / Melhorias

- [ ] Automação via GitHub Actions para sync programado
- [ ] Suporte a múltiplos datasets (ex: modelos NSFW separados)
- [ ] Cache de modelos entre runs do Kaggle Notebook
- [ ] Integração com ComfyUI-Manager para auto-install de missing nodes
- [ ] Monitoring de uso de VRAM/disk

---

## Licença

Uso interno - Automa / Vitor Ramos