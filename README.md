# Talegjenkjenningsdatasett fra videoer med undertitler
Dette er et repo med funksjonalitet for å lage talegjenkjenning-datasett (eng: automatic speech recognition, ASR) fra videoer med undertitler.

## Hvordan installere

### Installer ikke-python dependencies (kreves for BeautifulSoup og Pyenv) via apt:
```
sudo apt install \
    build-essential  \
    libffi-dev libxml2-dev libxslt1-dev  \
    curl  \
    libbz2-dev
```

### Installer [Rust](https://www.rust-lang.org/) toolchain med [rustup](https://rustup.rs/):
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Installer Pyenv
```
curl https://pyenv.run | bash
```

### Legg til Pyenv i path
```
echo "export PYENV_ROOT=\"$HOME/.pyenv\"" >> ~/.bashrc
echo "[[ -d \$PYENV_ROOT/bin ]] && export PATH=\"$PYENV_ROOT/bin:\$PATH\"" >> ~/.bashrc
echo "eval \"\$(pyenv init -)\"" >> ~/.bashrc

source ~/.bashrc
```

### Installer Python 3.11
```
pyenv install 3.11:latest
pyenv global 3.11.7
pyenv rehash
```
Bytt ut `3.11.7` med den versjonen som ble installert av `pyenv install 3.11:latest`

### Installer [pipx](https://pipx.pypa.io/stable/) (pip for executables)
```
pip install pipx
```

### Installer [PDM](https://pdm-project.org/latest/) (pakkeverktøy, litt som Poetry)
```
pipx install pdm
pipx ensurepath
```

### Installer [pre-commit](https://pre-commit.com/) (verktøy som sjekker kodekvalitet før commit)
```
pipx install pre-commit
pre-commit install
```

### Opprett virtuelt miljø med alle dependencies og utvikler-dependencies installert
```
pdm install --with dev
```


## Hent lyd og undertekster fra videofiler

```
pdm run python -m subtitled_videos_to_asr_dataset.audio_and_subtitle_extraction --source_dir <source-dir> --output_dir <output-dir> --languages <lang1> <lang2> ... --overwrite
```
### Forklaring
Hvor `<source-dir>` er stien til en mappe med video- og undertitteldata (mappe med undermapper av .mp4 og .vtt-filer)
og `<output-dir>` er stien til en mappe hvor lydfiler og undertekstfiler skal lagres (resultatet vil reflektere navn- og filstrukturen til source-dir).
Hvis `overwrite`, så vil videoer som allerede er gjort om til lyd og undertitler og kopiert over til output-dir hoppes over.
`languages` er en liste med språkkoder som kan brukes til å filtrere ut hvilke undertitler som hentes ut (default verdi er `["no", "nb", "nn"]`). Antar at undertittel-filene er navngitt `<språkode>.vtt`. Hvis det ikke finnes en undertittelfil for et av de gitte språkene, vil heller ikke lyden hentes ut.

## Prosesser undertekster og lyd
```
pdm run python -m subtitled_videos_to_asr_dataset.audio_and_subtitle_processing <input-directory> --output_directory <output-dir> --log_level <DEBUG/INFO/WARNING/ERROR/CRITICAL> --save_srt
```
### Forklaring
`output-dir` er et valgfritt argument og hvis det ikke spesifiseres, så blir output opprettet i input-directory.
`log-level` er et valgfritt argument for å angi loggnivået. Standard loggnivå er "INFO".
`save-srt` er et valgfritt argument og dersom det settes lagres også de sammenstilte undertekstene i srt format i tillegg til json.

## Lag audio-datasett
```
pdm run python -m subtitled_videos_to_asr_dataset.dataset_preparation.create_huggingface_dataset \
        --input_directory <input-dir> --output_directory <output-dir> --dataset_split_json <datasplit.json> \
        --langdetect_model <model-name> --langdetect_langcode <langcode> \
        --merge_n_seconds <N> --remove_longer_segments \
        --push --hf_repo_id <repo-id> --hf_private --hf_token <token> \
        --log_level <DEBUG/INFO/WARNING/ERROR/CRITICAL>
```
### Forklaring
`<input-dir>` er stien til en mappe med mapper med .mp3-filer og _aligned_with_language.json-filer (produsert av audio_and_subtitle_processing-modulen).
`<output-dir>` er stien til mappa der audio datasettet lages.
`--dataset_split_json` er et valgfritt argument. `<datasplit.json>` er en fil med train/val/test som keys og lister av submappe-navn fra `<input-dir>` som values.
`<model-name>` korresponderer til language identification (lid) modellen som er brukt for å detektere talespråk i lydsegmentene.
Det er bare segmenter hvor detektert språk er `<langcode>` som blir med i det ferdige datasettet.
`<N>` er makslengde (i sekunder) på segmenter. Kortere segmenter på rad vil om mulig limes sammen opp til `<N>`. Hvis `<N>` er 0 vil segmentene ikke limes sammen.
Hvis `--remove_longer_segments` er flagget, vil kun segmenter varighet `<N>` sekunder eller kortere bli med i det ferdige datasettet.
Hvis `--push` er flagget, vil datasettet pushes til huggingface.com.
`<repo-id>` er id-en til datasettrepoet som lages på huggingface.com (typisk `<profilnavn>/<datasettnavn>`)
Hvis `--hf_private` er flagget, vil datasettet lagres som et privat repo på huggingface.com.
`<token>` er et huggingface token med skrivetilgang til datasettrepoet.
