# Documentação — Treino YOLOv8 (dataset `geral-thermal-2024`) para DJI AI Inside

Registro do processo de adaptação, configuração e treino de um modelo YOLOv8 sobre o
MMYOLO (com o patch da DJI) para posterior quantização e execução embarcada em drone
DJI (Matrice 4 Series). Inclui os valores usados neste projeto e os problemas reais
encontrados durante a execução, com as respectivas correções.

- **Servidor:** `oper@asimov` (acesso via SSH)
- **Repositório MMYOLO:** `~/mmyolo_train/train4.0/mmyolo`
- **Ambiente conda:** `mmyolo`
- **Dataset:** `data/geral-thermal-2024/` (imagens térmicas, 5 classes, 6633 imagens de treino, 43795 anotações)

---

## 1. Estrutura do dataset

```
data/geral-thermal-2024/
├── train_coco.json      # anotações COCO (treino)
├── val_coco.json        # anotações COCO (validação)
├── test_coco.json       # anotações COCO (teste)
├── train/
│   ├── images/          # 0001.jpg, 0002.jpg, ...
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Categorias (na ordem do JSON):**

| id | classe |
|----|--------|
| 0 | Car |
| 1 | Person |
| 2 | Motocycle |
| 3 | Truck |
| 4 | Animal |

Comando usado para inspecionar o JSON:

```bash
python -c "
import json
d = json.load(open('data/geral-thermal-2024/train_coco.json'))
print('categories:', [(c['id'], c['name']) for c in d['categories']])
print('num annotations:', len(d['annotations']))
print('sample category_ids:', [a['category_id'] for a in d['annotations'][:10]])
"
```

Ponto importante: os `file_name` no JSON são apenas o nome do arquivo (`0001.jpg`), e as
imagens estão em `train/images/`. Por isso o prefixo de imagem é `train/images/` (e não `train/`).

---

## 2. Alterações no config

Arquivo: `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`
(único config permitido pela DJI; já vem modificado pelo patch `0001-NEW-ai-inside-init.patch`).

### 2.1 Bloco de dados e classes

```python
data_root = 'data/geral-thermal-2024/'      # ATENÇÃO: barra final obrigatória

train_ann_file = 'train_coco.json'
train_data_prefix = 'train/images/'

val_ann_file = 'val_coco.json'
val_data_prefix = 'val/images/'

num_classes = 5
class_name = ('Car', 'Person', 'Motocycle', 'Truck', 'Animal')
metainfo = dict(
    classes=class_name,
    palette=[(20, 220, 60), (0, 0, 255), (255, 0, 0), (0, 220, 220), (220, 20, 60)]
)
```

Regras:
- `num_classes` deve ser **menor que 10** (exigência da DJI). Aqui = 5.
- O número de cores da `palette` deve ser **igual** ao número de classes.
- A ordem de `class_name` deve bater exatamente com a ordem dos `categories` do JSON.

### 2.2 Adicionar `metainfo` aos dataloaders (passo mais fácil de esquecer)

Dentro de **`train_dataloader`** e **`val_dataloader`**, no `dataset=dict(...)`, adicionar
`metainfo=metainfo,` logo abaixo de `data_root=data_root,`:

```python
train_dataloader = dict(
    ...
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,                       # <-- ADICIONAR
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        ...
    )
)

val_dataloader = dict(
    ...
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,                       # <-- ADICIONAR
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        ...
    )
)
```

Sem essa linha, o dataset é carregado com as classes padrão do COCO, as anotações não
casam e o treino roda com **loss zero** (ver seção 5).

### 2.3 Parâmetros de treino (batch / épocas / learning rate)

Escolhas usadas neste projeto (6633 imagens, 1 GPU):

```python
train_batch_size_per_gpu = 16    # reduzir para 8/4 se der CUDA out of memory
max_epochs = 200                 # 500 (padrão COCO) é exagero para este dataset
base_lr = 0.0025                 # AJUSTADO: ver nota abaixo
```

**Nota sobre o learning rate:** o config vem com `base_lr = 0.01`, calibrado para
batch total de 64 (8 GPUs × 16). Com 1 GPU e batch 16, o batch total é 16 → o LR deve ser
escalado proporcionalmente (regra do *linear scaling*):

| batch total (1 GPU) | base_lr sugerido |
|---------------------|------------------|
| 16 | 0.0025 |
| 8  | 0.00125 |
| 4  | 0.000625 |

### 2.4 Avaliador (conferir)

```python
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,    # concatenação exige data_root com barra final
    metric='bbox')
test_evaluator = val_evaluator
```

Como usa `data_root + val_ann_file` (concatenação de strings), o `data_root` **precisa
terminar em `/`**, senão vira `data/geral-thermal-2024val_coco.json` (ver seção 5).

---

## 3. Verificação antes de treinar

Sempre validar a configuração antes de gastar GPU. O `browse_dataset.py` desenha as
imagens com as anotações lidas a partir do config:

```bash
python tools/analysis_tools/browse_dataset.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --out-dir /tmp/browse_check --not-show --show-number 20
```

Observações:
- O argumento correto nesta versão é `--out-dir` (e **não** `--output-dir`).
- `--show-number 20` gera só 20 imagens (mais rápido que o dataset inteiro).
- `--mode original` mostra as caixas na imagem crua; o padrão (`transformed`) mostra a
  imagem após augmentation (mosaico, resize).

Depois, copiar uma imagem para a máquina local (o servidor é headless) e conferir se as
bounding boxes aparecem com os rótulos corretos:

```bash
# rodar na máquina LOCAL, não no SSH:
scp oper@asimov:/tmp/browse_check/0001.jpg .
```

Avisos como `bbox is out of bounds`, `__floordiv__ is deprecated` ou
`Failed to add LocalVisBackend` são apenas *warnings* e não impedem o funcionamento.

---

## 4. Treino

Recomenda-se rodar dentro de `tmux` para o treino sobreviver a quedas da conexão SSH:

```bash
tmux new -s treino
```

Comando de treino (1 GPU):

```bash
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py 1
```

Controles do tmux: desanexar com `Ctrl+B` depois `D`; voltar com `tmux attach -t treino`.

**Sinal de treino saudável:** nas primeiras iterações o `loss` aparece **positivo**
(tipicamente entre ~2 e ~5) e vai diminuindo. Se aparecer `loss: 0.0000`, algo está
errado (ver seção 5).

As saídas (checkpoints `.pth`, logs) ficam em:

```
work_dirs/yolov8_s_syncbn_fast_8xb16-500e_coco/
```

Para limpar saídas de execuções anteriores:

```bash
rm -rf work_dirs/yolov8_s_syncbn_fast_8xb16-500e_coco
```

---

## 5. Problemas encontrados e soluções

Registro dos erros reais deste projeto, para referência futura.

### 5.1 Pacote PyTorch corrompido ao criar o ambiente conda

**Sintoma:**
```
CondaVerificationError: The package for pytorch ... appears to be corrupted.
The path '.../torch/version.py' specified in the package manifest cannot be found.
```

**Causa:** download/extração incompleta do pacote no cache do conda (rede instável ou
disco cheio). O conda reaproveita o pacote quebrado em `pkgs/`.

**Solução:** remover a pasta órfã do ambiente e o pacote do cache, depois recriar.
```bash
rm -rf /home/oper/miniconda3/envs/mmyolo
rm -rf /home/oper/miniconda3/pkgs/pytorch-1.10.1-py3.8_cuda11.3_cudnn8.2.0_0*
conda clean --all -y
conda create -n mmyolo python=3.8 -y
conda activate mmyolo
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
```
Verificar espaço com `df -h /home` — disco cheio causa corrupção recorrente no mesmo arquivo.

### 5.2 `FileNotFoundError` no arquivo de validação

**Sintoma:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/geral-thermal-2024val_coco.json'
```

**Causa:** faltou a barra final em `data_root`. O `val_evaluator` usa
`data_root + val_ann_file` (concatenação pura), gerando o caminho grudado.

**Solução:**
```python
data_root = 'data/geral-thermal-2024/'   # com a barra final
```

### 5.3 Loss zerado desde o início (`loss: 0.0000`)

**Sintoma:** todas as perdas ficam em `0.0000` (`loss_cls`, `loss_bbox`, `loss_dfl`),
`grad_norm` minúsculo, e na validação:
```
ERROR - The testing results of the whole dataset is empty.
```

**Causa:** o `metainfo` não foi adicionado aos dataloaders (só estava definido no topo do
config). Sem ele, o dataset carrega as classes do COCO em vez das 5 do projeto, as
anotações são descartadas e o modelo treina sem alvos.

**Diagnóstico rápido:**
```bash
grep -n "metainfo" configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
```
O `metainfo` deve aparecer **3 vezes**: definição + `train_dataloader` + `val_dataloader`.
Se aparecer só 1, é essa a causa.

**Solução:** adicionar `metainfo=metainfo,` nos dois dataloaders (ver 2.2).

### 5.4 Erros de grafia no config

Erros observados que quebram silenciosamente o `metainfo`:
- `classe_name` em vez de `class_name`.
- `pallete` em vez de `palette`.

Sempre conferir com:
```bash
grep -n "class_name\|classe_name\|palette\|pallete\|metainfo\|num_classes" \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
```

### 5.5 `--output-dir` não reconhecido no browse_dataset

**Sintoma:**
```
browse_dataset.py: error: unrecognized arguments: --output-dir /tmp/browse_check
```
**Solução:** o argumento correto é `--out-dir`.

### 5.6 `KeyboardInterrupt` no browse_dataset

Não é erro — é o `Ctrl+C` do usuário interrompendo a geração das imagens de propósito
(não é necessário gerar as 6633; algumas dezenas bastam para conferir).

---

## 6. Próximas etapas (pós-treino)

1. **Selecionar o melhor checkpoint** `.pth` em `work_dirs/...` (acompanhar o mAP na
   validação e escolher o de melhor desempenho).
2. **Reunir os artefatos para a DJI:**
   - o arquivo `.pth`;
   - 500 a 1000 imagens de calibração para quantização (ex.: `val/images/` em `.zip`);
   - os parâmetros de saída (`model_test_cfg`): `multi_label=True`, `nms_pre=30000`,
     `score_thr=0.001`, `nms.iou_threshold=0.7`, `max_per_img=300`.
3. **Upload e quantização:** `https://developer.dji.com/ai-inside/model` → *Upload Model*.
4. **Validação** na plataforma AI Inside.
5. **Distribuição:** `https://developer.dji.com/ai-inside/distribution` → gerar `model.zip`
   por SN de aeronave → copiar para o cartão SD.
6. **Uso no drone:** app DJI Pilot 2 → *Algorithm Management* → *From SD Card* → selecionar
   o modelo → ativar *AI* na visualização da câmera.

> Observação sobre 4K: para rodar o modelo em resolução 4K, alterar `widen_factor` de
> `0.5` para `0.25` **antes** do treino, senão a quantização falha na calibração.

---

## 7. Referência rápida de comandos

```bash
# ativar ambiente
conda activate mmyolo
cd ~/mmyolo_train/train4.0/mmyolo

# inspecionar categorias do dataset
python -c "import json; d=json.load(open('data/geral-thermal-2024/train_coco.json')); print([(c['id'],c['name']) for c in d['categories']], len(d['annotations']))"

# conferir grafia/ocorrências no config
grep -n "class_name\|palette\|metainfo\|num_classes\|data_root" \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py

# validar visualmente (20 imagens)
python tools/analysis_tools/browse_dataset.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --out-dir /tmp/browse_check --not-show --show-number 20

# treinar (dentro de tmux)
tmux new -s treino
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py 1

# limpar saídas anteriores
rm -rf work_dirs/yolov8_s_syncbn_fast_8xb16-500e_coco
```
