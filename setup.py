"""构建期把原生库编好放进 wheel.

装包后无需本机 C 编译器即可使用 C 内核; 构建机上没有编译器时跳过, 行为与
纯源码包一致 (运行期回退源码编译)。

含预编译库时 wheel 打平台专属标签 (py3-none-win_amd64 等), 避免把 A 平台的
二进制发给 B 平台。平台不匹配时 pip 从 sdist 构建, 源码路径仍完整可用。

产物落在 sunnynet/_prebuilt/<平台标签>/, 运行期按当前平台标签查找。
"""
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent


def _prebuilt_platform():
    sys.path.insert(0, str(ROOT))
    try:
        from sunnynet import _native
    except Exception:
        return None
    return _native.platform_tag()


def _has_prebuilt(*bases) -> bool:
    tag = _prebuilt_platform()
    if tag is None:
        return False
    for base in bases:
        if not base:
            continue
        d = Path(base) / "_prebuilt" / tag
        if d.is_dir() and any(d.iterdir()):
            return True
    return False


class build_py(_build_py):
    def run(self):
        super().run()
        sys.path.insert(0, str(ROOT))
        try:
            from sunnynet import _native
        except Exception as e:
            print(f"[prebuilt] 导入 _native 失败, 跳过: {type(e).__name__}: {e}")
            return
        try:
            made, err = _native.build_prebuilt()
        except Exception as e:
            print(f"[prebuilt] 编译异常, 跳过: {type(e).__name__}: {e}")
            return
        if err and not made:
            print(f"[prebuilt] 未生成预编译库: {err}")
            return
        if err:
            print(f"[prebuilt] 部分库未生成: {err}")
        dest = (Path(self.build_lib) / "sunnynet" / "_prebuilt"
                / _native.platform_tag())
        dest.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, dest / p.name)
        print(f"[prebuilt] {len(made)} 个库 -> {dest}")


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    _bdist_wheel = None


if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            if _has_prebuilt(ROOT / "sunnynet"):
                self.root_is_pure = False

        def get_tag(self):
            # build 已执行, 此处以 build_lib 为准 (CI 上源码目录没有产物)
            if _has_prebuilt(getattr(self, "build_lib", None),
                             ROOT / "sunnynet"):
                _, _, plat = super().get_tag()
                return "py3", "none", plat
            return super().get_tag()

    _cmdclass = {"build_py": build_py, "bdist_wheel": bdist_wheel}
else:
    _cmdclass = {"build_py": build_py}

setup(cmdclass=_cmdclass)