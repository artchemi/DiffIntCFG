# DiffInt: A Diffusion Model for Structure-Based Drug Design with Explicit Hydrogen Bond Interaction Guidance

Оригинальная модель описана в [статье](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01385). 

## 🛠 Зависимости

**1. Установка окружения.**

```bash
mamba env create -n Int-env-main -f environment_cfg.yml
```

Если будут конфликты с зависимостями при установке окружения из `environment_cfg.yml`, то можно установить базовое окружение из `environment.yml` и установить библиотеки в ручном режиме.

**2. Датасет с [Google Drive](https://drive.google.com/file/d/1RwDXBRVLRcEjSNHTw1JG6TpNgNUIogX2/view).**

```bash
cd data/
tar xvzf DiffInt_crossdock_data.tar.gz   
```

**3. Настройка конфига и запуск обучения.** `DiffInt_ca_double_src.yml` - конфиг из начального репозитория. `DiffInt_ca_double.yml` - конфиг для запуска на локальной машине. Запуск обучения:

```bash
python train.py --config configs/DiffInt_ca_double.yml
```

Продолжение обучение с определенного чекпоинта:
```bash
python train.py --config configs/DiffInt_ca_double.yml --resume logs/DiffInt_training/checkpoints/last.ckpt
```
