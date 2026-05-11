image2BVH portable
==================

1 枚の画像から人物を抽出 → 全身ポーズを推定 → BVH モーションファイルとして
書き出すスタンドアロン WebUI です。Windows 10 / 11 (x64) で動作します。

English: README.en.txt


使い方
------

詳しい使い方・作例は下記の記事をご覧ください。

    https://note.com/tori29umai/n/n66e124812f69


インストール
------------

1. image2bvh-__VERSION__-portable.exe を、アプリを置きたい場所
   (例: C:\Users\<あなたのユーザー名>\Apps\ など書き込み可能なフォルダ)
   にコピーします。

2. .exe をダブルクリック → 確認ダイアログで「はい」をクリック。

3. .exe と同じフォルダの直下に image2bvh\ が作られ、約 9 GB のファイルが
   解凍されます (数分)。

解凍先は .exe を置いたフォルダの直下に固定されます。別の場所にインストール
したい場合は、.exe を希望のフォルダへ先に移動してから実行してください。


起動
----

解凍後に作られた image2bvh\image2bvh.exe をダブルクリック。

数十秒で WebUI がブラウザで http://127.0.0.1:7860 に開きます。
初回起動のみ MHR 静止骨格のベイクと Triton JIT コンパイルで追加 30 秒〜
1 分ほどかかります (2 回目以降はキャッシュされて高速)。


動作環境
--------

OS:       Windows 10 / 11 x64

GPU:      NVIDIA RTX シリーズ推奨 (ドライバ R580 以降)
          - CUDA Toolkit のインストールは不要 (同梱の torch wheel が
            CUDA 13.0 ランタイム DLL を内蔵しています)
          - GPU が無い / 互換ドライバが無い場合は自動的に CPU 推論に
            フォールバックします (数倍遅くなります)

ディスク: 解凍後 約 10 GB の空き


アンインストール
----------------

image2bvh\ フォルダを削除するだけ。

このアプリは Windows レジストリを一切書き換えません。設定 (config.ini)、
モデルキャッシュ、一時 BVH 出力 (tmp\)、Triton JIT キャッシュ
(triton-cache\) など、すべての状態が image2bvh\ フォルダ内で完結します。
フォルダごと消せばマシンに痕跡は残りません。


トラブルシューティング
----------------------

* 初回起動が異常に遅い
    MHR 静止骨格のベイク (初回限定、runtime\mhr_rest.json にキャッシュ)。
    通常 30 秒〜1 分で完了。

* 「GPU を認識しない」「CPU でしか動かない」
    NVIDIA ドライバが R580 以降か確認 (nvidia-smi で確認可能)。

* GPU 推論中に Out of Memory
    起動前に環境変数 CUDA_VISIBLE_DEVICES= (空文字) を設定すると CPU
    モードに固定できます。

* ブラウザが自動で開かない
    手動で http://127.0.0.1:7860 を開いてください。

* Windows Defender / SmartScreen で警告
    自己署名されていない PyInstaller 製の単一 EXE のため誤検知される場合が
    あります。配布元を確認の上で「詳細情報」→「実行」を選択してください。


ライセンス
----------

このアプリは Meta の SAM 3 / SAM 3D Body / DINOv3 モデルを再配布しています。
利用 = 以下のライセンスに同意とみなされます。

  * SAM License (Meta SAM 3 / SAM 3D Body)
  * DINOv3 License (Meta DINOv3)
  * image2BVH 自体の MIT License

各ライセンスの全文は LICENSE_BUNDLE.txt に同梱されています。

軍事・武器開発・核・諜報・ITAR 規制対象用途は禁止されています (SAM /
DINOv3 License §1.b.v)。商用利用・大規模配布の前にライセンス全文を確認
してください。
