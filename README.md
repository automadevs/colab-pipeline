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
                                                ┌─────────────────────┐
                                                │   SELEÇÃO MANUAL    │
                                                │  (ipywidgets 05/08) │
                                                └─────────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   GOOGLE DRIVE  │◀───▶│  KAGGLE NOTEBOOK │◀───▶│    KAGGLE SSD       │
│ (Persistência & │     │   (Compute/GPU)  │     │   (/kaggle/working/ │
│  Backup Manual) │     └──────────────────┘     │   ComfyUI/models)   │
└─────────────────┘               │              └─────────────────────┘
                                  ▼
                         ┌─────────────────┐
                         │     COMFYUI     │
                         │   (SSD Local)   │
                         │  output:        │
                         │  /kaggle/       │
                         │  working/       │
                         │  ComfyUI/output │
                         └─────────────────┘
```

**Separação de responsabilidades:**
- **Civitai** → Origem dos modelos
- **Google Colab** → Estação de transferência (Civitai → Kaggle Dataset)
- **Kaggle Dataset** → Armazenamento persistente de modelos core (`automamermaid/comfydocs`) — checkpoints, LoRAs, VAEs, text_encoders, controlnet, etc.
- **GitHub** → Código e scripts (`automadevs/colab-pipeline`) — clonado e atualizado dinamicamente no runtime
- **Kaggle Notebook (SSD Local)** → Runtime e geração:
  - Modelos selecionados manualmente vão para `/kaggle/working/ComfyUI/models`
  - Geração do ComfyUI ocorre **exclusivamente no SSD local**: `/kaggle/working/ComfyUI/output`
- **Google Drive** → Camada de **persistência, backup e sincronização sob demanda**:
  - `Automa/ComfyUI/outputs/`
  - `Automa/ComfyUI/workflows/`
  - `Automa/ComfyUI/logs/`
  - `Automa/ComfyUI/metadata/`
  - **O Google Drive NÃO fica no caminho crítico da geração e nunca é passado como `--output-directory` do ComfyUI.**

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
│   ├── 05_download_to_local.ipynb  # Seleção manual de modelos do Dataset → SSD
│   ├── 06_comfyui_setup.ipynb      # GitHub → ComfyUI local + custom nodes
│   ├── 07_sync_robust.ipynb        # Sync seletivo Dataset → SSD (idempotente)
│   ├── 08_master_pipeline.ipynb    # Orquestrador completo (clone, GPU, ComfyUI SSD local, seleção de modelos, health check)
│   └── 09_sync_outputs.ipynb       # Painel de sincronização manual com Google Drive
│
├── scripts/                  # Scripts Python compartilhados
│   ├── civitai_download.py   # Download robusto da Civitai
│   ├── kaggle_upload.py      # Upload para Kaggle Dataset (CLI + kagglehub)
│   ├── kaggle_sync.py        # Sync seletivo com exibição de tamanho formatado
│   ├── kaggle_drive_sync.py  # Sincronização idempotente streaming SHA-256 com Drive
│   ├── comfyui_setup.py      # Instalação ComfyUI e start com output local no SSD
│   └── gpu_detect.py         # Detecção e validação de GPU NVIDIA
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

## Fluxo 2: Iniciar ComfyUI e Gerar (Kaggle Notebook)

**Executar no Kaggle Notebook (com GPU):**

```bash
# Pipeline completo automatizado (recomendado):
kaggle_runtime/08_master_pipeline.ipynb
```

O `08_master_pipeline.ipynb`:
1. Clona/atualiza `automadevs/colab-pipeline` diretamente do GitHub.
2. Detecta e valida GPU NVIDIA e VRAM disponível.
3. Testa conexão com o Google Drive para persistência/backup.
4. Instala ComfyUI e custom nodes com output configurado no SSD local (`/kaggle/working/ComfyUI/output`).
5. Abre interface interativa (`ipywidgets.SelectMultiple`) mostrando nome, categoria e tamanho formatado de cada modelo.
6. Baixa somente os modelos selecionados para o SSD local.
7. Inicia ComfyUI em background gerando no SSD local e valida via health check (`:8188/system_stats`).
8. Oferece push inicial condicional de outputs/logs se já existirem arquivos locais.

---

## Fluxo 3: Sincronização Manual com Google Drive

Para fazer backup ou restaurar arquivos entre o SSD local e o Google Drive, abra o painel manual:

```bash
kaggle_runtime/09_sync_outputs.ipynb
```

O notebook disponibiliza células independentes para:
- **Testar Conexão** com o Drive
- **Push Outputs**: `/kaggle/working/ComfyUI/output` → Drive `outputs/`
- **Push Workflows**: Workflows `.json` locais → Drive `workflows/`
- **Push Logs**: Logs locais (`comfyui.log`) → Drive `logs/`
- **Push Metadata**: Metadados (se existirem) → Drive `metadata/`
- **Pull Workflows**: Drive `workflows/` → local
- **Pull Outputs** (opcional): Drive `outputs/` → local
- **Pull Logs** (opcional): Drive `logs/` → local
- **Pull Metadata** (opcional): Drive `metadata/` → local

---

## Scripts Compartilhados (CLI)

Todos os notebooks usam scripts em `scripts/` para lógica reutilizável:

```bash
# Download Civitai
python scripts/civitai_download.py   --model-name "lustifyNSFWCheckpoint_v10Krea2.safetensors"   --staging-dir /content/kaggle_staging

# Upload Kaggle Dataset
python scripts/kaggle_upload.py   --model-name "lustifyNSFWCheckpoint_v10Krea2.safetensors"   --dataset "automamermaid/comfydocs"   --method cli  # ou kagglehub

# Sync seletivo Kaggle Dataset → SSD Local (apenas modelos desejados)
python scripts/kaggle_sync.py   --dataset "automamermaid/comfydocs"   --target-dir /kaggle/working/ComfyUI/models   --categories checkpoints loras vae

# Setup ComfyUI (SSD Local)
python scripts/comfyui_setup.py   --comfyui-dir /kaggle/working/ComfyUI   --output-dir /kaggle/working/ComfyUI/output   --custom-nodes "ltdrdata/ComfyUI-Manager" "cubiq/ComfyUI_essentials"   --start --health-check

# Detectar GPU
python scripts/gpu_detect.py --recommend

# Sync com Google Drive (idempotente via streaming SHA-256)
python scripts/kaggle_drive_sync.py   --action push   --categories outputs workflows logs   --drive-base "Automa/ComfyUI"   --env kaggle

# Pull workflows do Drive
python scripts/kaggle_drive_sync.py   --action pull   --categories workflows   --drive-base "Automa/ComfyUI"   --env kaggle
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
2. Ative **GPU** (Settings → Accelerator → GPU T4 x2 ou P100)
3. Configure **Secrets** (ícone de chave):
   - `KAGGLE_USERNAME` + `KAGGLE_KEY` (para listar e baixar modelos do dataset)
   - `GDRIVE_SERVICE_ACCOUNT_JSON`: Service Account JSON com acesso ao Google Drive (para sync e backup)
4. Execute `08_master_pipeline.ipynb` (o repositório será clonado automaticamente)

### Google Drive (Persistência & Backup)

**Estrutura no Drive:**
```
/Meu Drive/Automa/ComfyUI/
├── outputs/          # Imagens e vídeos gerados (sincronizados sob demanda)
├── workflows/        # Workflows salvos (.json)
├── logs/             # Logs de execução do ComfyUI
└── metadata/         # Metadados e registros estruturados
```

**Configuração do Service Account (Kaggle):**
1. No Google Cloud Console: crie Service Account → Role: Editor → Create Key (JSON)
2. Compartilhe a pasta `Automa/ComfyUI` no Google Drive com o email do Service Account (Editor)
3. No Kaggle: Secrets → `GDRIVE_SERVICE_ACCOUNT_JSON` = conteúdo do JSON

**No Colab:** Usa `google.colab.drive.mount()` nativo (sem service account).

---

## Categorias de Modelos Suportadas

O Dataset `automamermaid/comfydocs` armazena os modelos por categoria:

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

## Troubleshooting

| Problema | Causa | Solução |
|---|---|---|
| Kaggle 403 no upload | Sem permissão de edição | Verifique se é owner/colaborador do dataset `automamermaid/comfydocs` |
| Download 0 bytes | Token Civitai inválido | Verifique `CIVITAI_TOKEN` nos Secrets |
| Modelo não encontrado no dataset | Upload anterior falhou | Re-execute upload (03 ou 04) |
| Tamanho divergente | Download parcial | Delete staging e rebaixe |
| ComfyUI não inicia | Dependências faltando | Execute `06_comfyui_setup.ipynb` novamente |
| VRAM insuficiente | Modelo muito grande | Use `--lowvram` ou `--cpu` no ComfyUI |
| Drive não monta (Kaggle) | Service Account inválido | Verifique `GDRIVE_SERVICE_ACCOUNT_JSON` nos Secrets |
| Drive permission denied | Pasta não compartilhada | Compartilhe `Automa/ComfyUI` com o email do Service Account (Editor) |
| rclone não encontrado | Pacote ausente | Instale via apt ou pip |
| Sync Drive pulou arquivo | Arquivo já idêntico | Comportamento correto: mesmo tamanho e hash SHA-256 são preservados |

---

## Checklist de Validação

### Colab Transfer
- [ ] Python 3.10+ no Colab
- [ ] Kaggle CLI ≥ 1.5.12
- [ ] `~/.kaggle/kaggle.json` configurado
- [ ] `CIVITAI_TOKEN` nos Secrets
- [ ] Dataset `automamermaid/comfydocs` acessível
- [ ] Permissão de edição no dataset
- [ ] Upload cria nova versão no Kaggle (sem erro 403)

### Kaggle Runtime
- [ ] GPU ativada (T4, P100, ou A100)
- [ ] `kaggle.json` nos Secrets
- [ ] `GDRIVE_SERVICE_ACCOUNT_JSON` nos Secrets
- [ ] Repositório clonado e scripts atualizados
- [ ] ComfyUI instalado no SSD local
- [ ] Modelos selecionados sincronizados do Dataset → SSD
- [ ] ComfyUI inicia gerando em `/kaggle/working/ComfyUI/output`
- [ ] ComfyUI responde em `:8188` ao health check
- [ ] Sync com Google Drive executado sob demanda via `09_sync_outputs.ipynb`

---

## Licença

Uso interno - Automa / Vitor Ramos\n