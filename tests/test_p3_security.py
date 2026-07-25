"""
P3 安全测试：验证 Docker 镜像依赖版本固定与供应链安全修复。

覆盖范围：
1. Dockerfile 基础镜像固定到具体 patch 版本（非浮动 tag）
2. Dockerfile 多阶段构建（builder + runtime）
3. Dockerfile pip 安装使用 --no-cache-dir 防止缓存层泄露
4. Dockerfile 以非 root 用户运行
5. requirements.lock 完全固定依赖版本（== 精确版本，无 >= 或 ~=）
6. requirements.lock 不含可编辑安装行（-e .）和 Windows 路径
7. pyproject.toml 使用 ~= 兼容版本约束（非 >= 松散约束）
8. Dockerfile 使用 requirements.lock 而非 requirements.txt
"""

import re
from pathlib import Path

import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
REQUIREMENTS_LOCK = PROJECT_ROOT / "requirements.lock"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS_TXT = PROJECT_ROOT / "requirements.txt"


@pytest.fixture(scope="module")
def dockerfile_content():
    """读取 Dockerfile 内容"""
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lock_content():
    """读取 requirements.lock 内容"""
    return REQUIREMENTS_LOCK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pyproject_content():
    """读取 pyproject.toml 内容"""
    return PYPROJECT_TOML.read_text(encoding="utf-8")


class TestP3BaseImagePin:
    """P3：基础镜像版本固定"""

    def test_no_floating_tag_latest(self, dockerfile_content):
        """Dockerfile 不应使用 :latest 浮动 tag"""
        assert ":latest" not in dockerfile_content, "Dockerfile 不应使用 :latest 浮动 tag"

    def test_base_image_pinned_to_patch_version(self, dockerfile_content):
        """基础镜像应固定到具体 patch 版本（如 python:3.13.7-slim），非 python:3 或 python:3-slim"""
        from_lines = re.findall(r"^FROM\s+(\S+)", dockerfile_content, re.MULTILINE)
        assert len(from_lines) >= 2, "应使用多阶段构建（至少 2 个 FROM）"
        for image_ref in from_lines:
            # 跳过 AS builder/runtime 别名部分
            image = image_ref.split(" AS ")[0].strip()
            # 提取 tag 部分
            if ":" in image:
                tag = image.split(":")[-1]
                # tag 应包含完整的 patch 版本号（如 3.13.7-slim），不应是 3 或 3-slim
                assert re.search(r"\d+\.\d+\.\d+", tag), f"基础镜像 {image} 的 tag '{tag}' 未固定到 patch 版本"

    def test_both_stages_use_same_pinned_version(self, dockerfile_content):
        """builder 和 runtime 阶段应使用相同的基础镜像版本"""
        from_lines = re.findall(r"^FROM\s+(\S+)", dockerfile_content, re.MULTILINE)
        images = [line.split(" AS ")[0].strip() for line in from_lines]
        # 所有阶段应使用相同的镜像版本
        unique_images = set(images)
        assert len(unique_images) == 1, f"多阶段构建应使用相同基础镜像，实际：{unique_images}"


class TestP3MultiStageBuild:
    """P3：多阶段构建安全"""

    def test_has_builder_stage(self, dockerfile_content):
        """应有 builder 构建阶段"""
        assert "AS builder" in dockerfile_content, "缺少 'AS builder' 构建阶段"

    def test_has_runtime_stage(self, dockerfile_content):
        """应有 runtime 运行阶段"""
        assert "AS runtime" in dockerfile_content, "缺少 'AS runtime' 运行阶段"

    def test_no_build_tools_in_runtime(self, dockerfile_content):
        """runtime 阶段不应安装 gcc 等构建工具"""
        # 找到 runtime 阶段开始位置
        runtime_split = dockerfile_content.split("AS runtime")
        assert len(runtime_split) == 2, "应只有一个 runtime 阶段"
        runtime_section = runtime_split[1]
        # runtime 阶段不应有 apt-get install gcc
        assert "gcc" not in runtime_section, "runtime 阶段不应包含 gcc 构建工具"

    def test_no_cache_dir_in_pip(self, dockerfile_content):
        """pip install 应使用 --no-cache-dir 防止缓存泄露"""
        # 仅检查实际命令行（非注释行）
        pip_install_lines = [
            line
            for line in dockerfile_content.splitlines()
            if "pip install" in line and not line.strip().startswith("#")
        ]
        assert len(pip_install_lines) > 0, "应至少有一条 pip install 命令"
        for line in pip_install_lines:
            if "--upgrade pip" not in line:
                assert "--no-cache-dir" in line, f"pip install 命令应包含 --no-cache-dir：{line.strip()}"


class TestP3NonRootUser:
    """P3：非 root 用户运行"""

    def test_creates_non_root_user(self, dockerfile_content):
        """应创建非 root 用户"""
        assert "useradd" in dockerfile_content, "应使用 useradd 创建非 root 用户"

    def test_user_directive_present(self, dockerfile_content):
        """应有 USER 指令切换到非 root 用户"""
        assert re.search(r"^USER\s+\S+", dockerfile_content, re.MULTILINE), "缺少 USER 指令，容器应以非 root 用户运行"

    def test_user_is_not_root(self, dockerfile_content):
        """USER 指令不应是 root"""
        user_match = re.search(r"^USER\s+(\S+)", dockerfile_content, re.MULTILINE)
        assert user_match, "缺少 USER 指令"
        username = user_match.group(1)
        assert username != "root", "容器不应以 root 用户运行"

    def test_shell_disabled_for_appuser(self, dockerfile_content):
        """非 root 用户应禁用登录 shell（--shell /bin/false）"""
        assert "/bin/false" in dockerfile_content, "非 root 用户应使用 --shell /bin/false 禁用登录"


class TestP3RequirementsLock:
    """P3：依赖锁文件完全固定版本"""

    def test_lock_file_exists(self):
        """requirements.lock 文件应存在"""
        assert REQUIREMENTS_LOCK.exists(), "requirements.lock 文件不存在"

    def test_all_versions_are_exact(self, lock_content):
        """锁文件中所有依赖应使用 == 精确版本（非 >= 或 ~=）"""
        # 跳过注释行
        dep_lines = [
            line.strip() for line in lock_content.splitlines() if line.strip() and not line.strip().startswith("#")
        ]
        assert len(dep_lines) > 50, f"锁文件依赖数量过少（{len(dep_lines)} 行），可能未正确生成"
        for line in dep_lines:
            assert "==" in line, f"依赖行应使用 == 精确版本：{line}"
            assert ">=" not in line, f"锁文件不应使用 >= 松散约束：{line}"
            assert "~=" not in line, f"锁文件不应使用 ~= 约束：{line}"

    def test_no_editable_install(self, lock_content):
        """锁文件不应包含可编辑安装行（-e .）"""
        # 仅检查非注释行
        for line in lock_content.splitlines():
            if not line.strip().startswith("#"):
                assert "-e " not in line, f"锁文件不应包含可编辑安装行：{line.strip()}"

    def test_no_windows_paths(self, lock_content):
        """锁文件不应包含 Windows 路径（如 d:\\ 或 C:\\）"""
        # 跳过注释行
        non_comment = [line for line in lock_content.splitlines() if not line.strip().startswith("#")]
        for line in non_comment:
            assert not re.search(r"[A-Za-z]:[\\/]", line), f"锁文件不应包含 Windows 路径：{line}"

    def test_no_dev_only_packages(self, lock_content):
        """锁文件不应包含 dev 专用工具（pytest/coverage/build 等）"""
        dev_packages = ["pytest", "pytest-cov", "coverage", "pip-tools", "build==", "wheel=="]
        for pkg in dev_packages:
            # 精确匹配包名行（不以 # 开头）
            for line in lock_content.splitlines():
                if not line.strip().startswith("#"):
                    assert not line.strip().startswith(pkg), f"锁文件不应包含 dev 专用包：{line.strip()}"

    def test_key_runtime_deps_present(self, lock_content):
        """锁文件应包含关键运行时依赖"""
        key_deps = [
            "langchain==",
            "langgraph==",
            "chromadb==",
            "fastapi==",
            "uvicorn==",
            "pydantic==",
            "streamlit==",
        ]
        for dep in key_deps:
            assert dep in lock_content, f"锁文件缺少关键运行时依赖：{dep}"


class TestP3PyprojectConstraints:
    """P3：pyproject.toml 版本约束安全"""

    def test_no_loose_ge_constraints(self, pyproject_content):
        """pyproject.toml 运行时依赖不应使用 >= 松散约束"""
        # 提取 dependencies 列表部分
        deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_content, re.DOTALL)
        assert deps_match, "未找到 dependencies 列表"
        deps_section = deps_match.group(1)
        # 检查每一行依赖声明
        dep_lines = [line.strip() for line in deps_section.splitlines() if line.strip().startswith('"')]
        assert len(dep_lines) > 0, "dependencies 列表为空"
        for line in dep_lines:
            # 跳过纯注释行
            if line.startswith("#"):
                continue
            # 提取引号内的依赖声明
            dep_str = re.search(r'"([^"]+)"', line)
            if dep_str:
                dep = dep_str.group(1)
                assert ">=" not in dep, f"依赖不应使用 >= 松散约束：{dep}"
                assert ">" not in dep or "~>" in dep, f"依赖不应使用 > 松散约束：{dep}"

    def test_uses_compatible_release(self, pyproject_content):
        """pyproject.toml 应使用 ~= 兼容版本约束"""
        deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_content, re.DOTALL)
        assert deps_match, "未找到 dependencies 列表"
        deps_section = deps_match.group(1)
        dep_lines = [
            re.search(r'"([^"]+)"', line).group(1)
            for line in deps_section.splitlines()
            if line.strip().startswith('"') and re.search(r'"([^"]+)"', line)
        ]
        tilde_count = sum(1 for d in dep_lines if "~=" in d)
        assert tilde_count == len(
            dep_lines
        ), f"所有运行时依赖应使用 ~= 约束，{tilde_count}/{len(dep_lines)} 个使用了 ~="

    def test_constraints_aligned_with_lock(self, pyproject_content, lock_content):
        """pyproject.toml 的 ~= 约束应与 requirements.lock 中实际版本兼容"""
        deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_content, re.DOTALL)
        assert deps_match
        deps_section = deps_match.group(1)
        # 解析每个依赖的 ~= 约束
        for line in deps_section.splitlines():
            dep_str_match = re.search(r'"([^"]+~=([0-9.]+))"', line)
            if not dep_str_match:
                continue
            full_dep = dep_str_match.group(1)
            constraint_version = dep_str_match.group(2)
            # 提取包名
            pkg_name = full_dep.split("~=")[0].strip()
            # 从锁文件中查找该包的精确版本
            lock_match = re.search(rf"^{re.escape(pkg_name)}==([0-9.]+)", lock_content, re.MULTILINE)
            if lock_match:
                lock_version = lock_match.group(1)
                # ~=1.3.0 表示 >=1.3.0, <1.4.0；锁文件版本应在此范围内
                constraint_parts = constraint_version.split(".")
                lock_parts = lock_version.split(".")
                # 至少 major.minor 应匹配
                assert (
                    constraint_parts[0] == lock_parts[0]
                ), f"{pkg_name}: ~= {constraint_version} 与锁定版本 {lock_version} major 版本不匹配"
                assert (
                    constraint_parts[1] == lock_parts[1]
                ), f"{pkg_name}: ~= {constraint_version} 与锁定版本 {lock_version} minor 版本不匹配"


class TestP3DockerfileUsesLock:
    """P3：Dockerfile 使用锁文件而非 requirements.txt"""

    def test_dockerfile_copies_lock_file(self, dockerfile_content):
        """Dockerfile 应 COPY requirements.lock"""
        assert "requirements.lock" in dockerfile_content, "Dockerfile 应使用 requirements.lock 而非 requirements.txt"

    def test_dockerfile_installs_from_lock(self, dockerfile_content):
        """Dockerfile 应从 requirements.lock 安装依赖"""
        assert "-r requirements.lock" in dockerfile_content, "Dockerfile 应从 requirements.lock 安装依赖"

    def test_dockerfile_does_not_use_requirements_txt(self, dockerfile_content):
        """Dockerfile 不应直接使用 requirements.txt 安装依赖"""
        # 不应出现 pip install -r requirements.txt
        assert "pip install" in dockerfile_content
        assert (
            "-r requirements.txt" not in dockerfile_content
        ), "Dockerfile 不应从 requirements.txt 安装依赖（应使用 requirements.lock）"
