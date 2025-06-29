# latent_self.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['latent_self.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('data', 'data'),
    ],
    hiddenimports=[
        'models.encoders.pSp',
        'mediapipe.python._framework_bindings',
        'appdirs',
        'yaml',
        'werkzeug.security',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='latent-self',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
