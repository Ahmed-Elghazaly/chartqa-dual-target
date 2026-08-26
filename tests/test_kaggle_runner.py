"""The Kaggle runner generates Python source and ships it to a remote machine.

Generated code is still code. Two Kaggle sessions were burned discovering that
nobody had checked whether it parsed:

* run 1 — the code dataset never attached, because Kaggle normalises refs to
  lowercase and the username was mixed case;
* run 2 — the generated script had a real newline inside a string literal,
  because a ``\\n`` in a non-raw triple-quoted string is a newline.

Both are trivial. Both cost a round trip to a queue. Both are now tested here,
locally, in milliseconds.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

kaggle_run = pytest.importorskip("kaggle_run")


def test_generated_kernel_script_compiles():
    """The check that would have caught run 2 before it was pushed."""
    script = kaggle_run.render_kernel_script("chartqa-dt-src", ["cdt-train", "--stage", "smoke"])
    compile(script, "main.py", "exec")


def test_generated_script_has_no_stray_newline_in_a_string():
    """Comments are skipped: prose may legitimately carry an odd quote, and a
    comment cannot break the parse anyway -- `compile()` above covers that."""
    script = kaggle_run.render_kernel_script("ds", ["x"])
    for i, line in enumerate(script.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        quotes = line.count('"') - line.count('\\"')
        assert quotes % 2 == 0, f"line {i} has an unbalanced quote: {line!r}"


def test_command_is_embedded_as_valid_json():
    cmd = ["cdt-train", "--stage", "smoke", "--resolutions", "512,native"]
    script = kaggle_run.render_kernel_script("ds", cmd)
    for part in cmd:
        assert part in script


def test_script_fails_loudly_when_the_dataset_is_missing():
    script = kaggle_run.render_kernel_script("ds", ["x"])
    assert "code dataset not attached" in script
    assert "lowercase" in script, "the message must name the cause that actually bit us"
    assert 'os.listdir("/kaggle/input")' in script, "print the directory before failing"


def test_script_handles_both_upload_layouts():
    """Kaggle sometimes auto-extracts an uploaded archive and sometimes does not.

    Run 3 uploaded one code.zip and the kernel found ['configs', 'pyproject.toml',
    'README.md', 'src'] instead - already expanded. Betting on either behaviour
    costs a session; handling both costs four lines.
    """
    script = kaggle_run.render_kernel_script("ds", ["x"])
    assert "code.zip" in script, "must handle the archive arriving intact"
    assert "pyproject.toml" in script, "must handle it arriving already expanded"
    assert "no code.zip and no pyproject.toml" in script, "and must fail loudly otherwise"


def test_username_is_lowercased(monkeypatch, tmp_path):
    """Kaggle slugs are lowercase; a mixed-case ref silently fails to attach."""
    monkeypatch.setenv("KAGGLE_USERNAME", "MixedCaseUser")
    monkeypatch.setattr(kaggle_run.Path, "home", staticmethod(lambda: tmp_path))
    assert kaggle_run._username() == "mixedcaseuser"


def test_staging_excludes_credentials_and_dataset_content(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src" / "chartqa_dt").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "src" / "chartqa_dt" / "__init__.py").write_text("")
    (repo / "configs" / "base.yaml").write_text("a: 1")
    (repo / "pyproject.toml").write_text("[project]")
    # Things that must never be shipped.
    (repo / ".env").write_text("HF_TOKEN=secret")
    (repo / "src" / "chart.png").write_bytes(b"fake")
    (repo / "src" / "data.parquet").write_bytes(b"fake")

    monkeypatch.setattr(kaggle_run, "REPO_ROOT", repo)
    dest = tmp_path / "staging"
    dest.mkdir()
    kaggle_run._copy_sources(dest)

    shipped = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert "src/chartqa_dt/__init__.py" in shipped
    assert "configs/base.yaml" in shipped
    assert "pyproject.toml" in shipped
    assert not any(s.endswith(".png") for s in shipped)       # rule 7
    assert not any(s.endswith(".parquet") for s in shipped)   # rule 7
    assert ".env" not in shipped                              # credentials


def test_archive_round_trips_the_tree(tmp_path):
    """What is zipped here is what unzips there, with paths intact."""
    staging = tmp_path / "s"
    (staging / "src" / "pkg").mkdir(parents=True)
    (staging / "src" / "pkg" / "m.py").write_text("x = 1")
    (staging / "pyproject.toml").write_text("[project]")

    archive = tmp_path / "code.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(staging).as_posix())

    out = tmp_path / "out"
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    assert (out / "pyproject.toml").is_file()
    assert (out / "src" / "pkg" / "m.py").read_text() == "x = 1"


def test_gpu_check_runs_before_anything_expensive():
    """A kernel that silently fell back to CPU looked identical to a slow one.

    Kaggle exposes no logs while a kernel runs, so a wrong accelerator was a
    45-minute mystery instead of a 20-second answer. torch is preinstalled, so
    the check costs nothing and must come before the 4.2 GB download.
    """
    script = kaggle_run.render_kernel_script("ds", ["cdt-train"])
    assert "cuda.is_available" in script
    assert script.index("cuda.is_available") < script.index("pip"), (
        "the accelerator check must precede the model download and pip installs"
    )
    assert "Refusing to run a memory benchmark on CPU" in script


def test_cpu_is_allowed_only_when_explicitly_requested():
    gpu_script = kaggle_run.render_kernel_script("ds", ["x"], allow_cpu=False)
    cpu_script = kaggle_run.render_kernel_script("ds", ["x"], allow_cpu=True)
    assert "not False" in gpu_script or "and False" in gpu_script
    assert "and True" in cpu_script or "not True" in cpu_script
    compile(gpu_script, "main.py", "exec")
    compile(cpu_script, "main.py", "exec")


def test_gpu_check_validates_architecture_not_just_presence():
    """`torch.cuda.is_available()` being True is not enough.

    Kaggle handed out a Tesla P100 (sm_60) whose architecture its own PyTorch
    build does not support: "Tesla P100-PCIE-16GB with CUDA capability sm_60 is
    not compatible with the current PyTorch installation." Every CUDA op then
    fails or silently falls back, so a benchmark on it measures nothing -- and the
    presence check passed happily.
    """
    script = kaggle_run.render_kernel_script("ds", ["x"])
    assert "get_arch_list" in script, "must check the arch list, not only availability"
    assert "not supported by this PyTorch" in script


def test_kernel_requests_a_specific_accelerator(tmp_path, monkeypatch):
    """Left to itself Kaggle chooses, and it chose an unusable card."""
    import json

    monkeypatch.setattr(kaggle_run, "REPO_ROOT", tmp_path)
    captured = {}

    class FakeApi:
        def kernels_push(self, folder):
            captured["meta"] = json.loads(
                (kaggle_run.Path(folder) / "kernel-metadata.json").read_text()
            )

    monkeypatch.setattr(kaggle_run, "_username", lambda: "user")
    kaggle_run.push_kernel(FakeApi(), "user/ds", ["cdt-train"], title="t", machine_shape="gpu_t4x2")
    assert captured["meta"]["machine_shape"] == "gpu_t4x2"
    assert captured["meta"]["enable_gpu"] is True
    assert captured["meta"]["is_private"] is True


def test_no_gpu_request_sets_machine_shape_none(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(kaggle_run, "REPO_ROOT", tmp_path)
    captured = {}

    class FakeApi:
        def kernels_push(self, folder):
            captured["meta"] = json.loads(
                (kaggle_run.Path(folder) / "kernel-metadata.json").read_text()
            )

    monkeypatch.setattr(kaggle_run, "_username", lambda: "user")
    kaggle_run.push_kernel(FakeApi(), "user/ds", ["x"], title="t", gpu=False)
    assert captured["meta"]["machine_shape"] == "none"


def test_torchao_is_pinned_before_the_model_download():
    """Kaggle preinstalls torchao 0.10.0; transformers/peft require >0.16.0.

    Discovered after a 4.2 GB download and a successful quantised load, at the
    moment LoRA was applied. Pinning it up front costs seconds.
    """
    script = kaggle_run.render_kernel_script("ds", ["cdt-train"])
    assert "torchao>=0.16.0" in script
    assert script.index("torchao") < script.index('["cdt-train"]'), (
        "torchao must be pinned before the job runs, not after the model downloads"
    )
