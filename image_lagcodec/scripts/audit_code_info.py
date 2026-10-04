"""Does a level's code say anything about its children beyond their mean? Reads the .npz written by
audit_tf_vs_gen_timesteps (targets + real context codes) and fits simple predictors of each child's deviation from
the row mean, on train rows, scored on val rows (R^2; 0 = the code carries the mean only).
Usage: python3 -m image_lagcodec.scripts.audit_code_info FILE.npz [--levels 0,1,2]
"""
import argparse
import numpy as np


def r2(pred, y):
    return 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean(0, keepdims=True)) ** 2).sum())


def feats(code, resid, kind):
    if kind == "code":
        f = [code / 255.0]
    elif kind == "resid":
        f = [resid / 8.0]
    else:
        r = resid / 8.0
        f = [code / 255.0, r, r ** 2, r[:, [0]] * r[:, [1]], r[:, [1]] * r[:, [2]], r[:, [0]] * r[:, [2]], r ** 3]
    return np.concatenate(f + [np.ones((code.shape[0], 1))], axis=1)


def load(z, split, lvl):
    T, C = z[f"{split}_l{lvl}_target"].astype(np.float64), z[f"{split}_l{lvl}_ctx"].astype(np.float64)
    B, L, D = T.shape
    rs = L // C.shape[1]
    rows = T.reshape(B * C.shape[1], rs, D)
    mean = rows.mean(1)
    dev = (rows - mean[:, None, :]).reshape(rows.shape[0], rs * D)
    code = C.reshape(-1, D)
    return code, code - mean, dev, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    a = ap.parse_args()
    z = np.load(a.npz)
    for lvl in a.levels:
        ct, rt, yt, mt = load(z, "train", lvl)
        cv, rv, yv, mv = load(z, "val", lvl)
        print(f"[level {lvl}] rows train={ct.shape[0]} val={cv.shape[0]} | code - row mean: rms={np.sqrt((rt ** 2).mean()):.2f} "
              f"| child deviation from row mean: variance={float((yt ** 2).mean()):.2f} (= mse floor of any mean-only decoder)")
        for kind in ("code", "resid", "code+resid poly"):
            Xt, Xv = feats(ct, rt, kind), feats(cv, rv, kind)
            W = np.linalg.solve(Xt.T @ Xt + 1e-3 * np.eye(Xt.shape[1]), Xt.T @ yt)
            print(f"  ridge on {kind:16s}: R^2 train={r2(Xt @ W, yt):+.4f} val={r2(Xv @ W, yv):+.4f}")
        # nonparametric: mean deviation pattern per sign-bin of the residual triple
        b = lambda r: ((np.digitize(r, [-2.5, 2.5]) * np.array([1, 3, 9])).sum(1)).astype(int)
        bt, bv = b(rt), b(rv)
        table = np.stack([yt[bt == k].mean(0) if (bt == k).any() else np.zeros(yt.shape[1]) for k in range(27)])
        print(f"  per residual-bin mean (27 bins)  : R^2 train={r2(table[bt], yt):+.4f} val={r2(table[bv], yv):+.4f}")
        # how well does the code give the row mean itself
        print(f"  code vs true row mean: mse={float((rt ** 2).mean()):.2f} (train) {float((rv ** 2).mean()):.2f} (val)", flush=True)


if __name__ == "__main__":
    main()
