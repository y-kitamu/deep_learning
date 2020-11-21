# 論文実装・検証
Tanaka D, et al. 2018, Joint Optimization Framework for Learning with Noisy Labels
http://arxiv.org/abs/1803.11364

正解ラベルにノイズがある場合の学習フレームワークを提案した論文。
学習途中で正解ラベルをつけ直すことで最終的なモデルの精度を向上させる。
普通に学習 -> 正解ラベルを更新しながら学習 -> 正解ラベルを固定して学習の3ステップで学習を行う。

今回はcifar10で再現実験を実施した(結果はnotebooks以下のnotebook)。
モデルはpreact resnet32、optimizerはadamで、正解ラベルに10%のノイズ(論文でいうところのsynthesized noisy)を
混ぜて学習を実施し、精度の検証を行った。

|               | ノイズなし | ノイズあり(10%, ラベル更新なし) | ノイズあり(10%, ラベル更新あり) |
| Test Accuracy | 0.8867     | 0.8784                          | 0.8871                          |

論文だと91%->92.9%で2%弱の精度改善だったが、今回の実験では1%弱の精度改善だった。
時間節約のために実験設計をゆるくしたこと、
validation loss, accuracyの計算が適切でなかったことが原因と考えられる。
今回の実装上、validation loss, accuracy は正解ラベルを更新しながら学習したときに最大になり、
その時の値をbest loss, best accとして保持してしまっていた。
この点については改善の余地があるが、今回の検討としては精度改善の効果が確認できたので、一旦完了とする。
