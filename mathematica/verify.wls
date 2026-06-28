#!/usr/bin/env wolframscript
(* Decisive check: compact multiplicity-ESP orbital  ==  true d^n JK projection,
   at the determinant level.  Ratio det_compact/det_true must be z-INDEPENDENT. *)

Nn = 4;
others[i_] := Complement[Range[Nn], {i}];

(* compact form: z^(n+m) * e_n( {1/(z_i-z_j)} each repeated jk times ) *)
espMult[i_, n_, jk_] := Coefficient[
   Expand[Product[(1 + t/(z[i] - z[j]))^jk, {j, others[i]}]], t, n];
compact[i_, n_, m_, jk_] := z[i]^(n + m) * espMult[i, n, jk];

(* true LLL (Jain-Kamilla) projection on the disk:
   normal-order zbar to the left -> zbar -> d/dz, acting on z^(n+m) * Ji^jk,
   then divide by Ji^jk.  Ji = Prod_{j!=i}(z_i - z_j). *)
Ji[i_, jk_] := Product[(z[i] - z[j]), {j, others[i]}]^jk;
true[i_, n_, m_, jk_] := D[z[i]^(n + m)*Ji[i, jk], {z[i], n}]/Ji[i, jk];

detOf[orb_, cfg_, jk_] := Det[Table[
    orb[i, cfg[[a, 1]], cfg[[a, 2]], jk], {i, Nn}, {a, Length[cfg]}]];

configs = {
   {"C1: all LLL",          {{0, 0}, {0, 1}, {0, 2}, {0, 3}}},
   {"C2: 3 n=0, 1 n=1",     {{0, 0}, {0, 1}, {0, 2}, {1, 0}}},
   {"C3: span n=0,1,2",     {{0, 0}, {0, 1}, {1, 0}, {2, 0}}},
   {"C4: 2 n=0, 2 n=1",     {{0, 0}, {0, 1}, {1, 0}, {1, 1}}},
   {"GAPPY: n=2 but no n=1", {{0, 0}, {0, 1}, {0, 2}, {2, 0}}}
   };

ratioConst[cfg_, jk_] := Module[{dc, dt, r},
   dc = detOf[compact, cfg, jk];
   dt = detOf[true, cfg, jk];
   If[dt === 0, Return["det_true=0"]];
   r = Together[dc/dt];
   If[FreeQ[Simplify[r], z], Row[{"CONST=", Simplify[r]}], "z-DEPENDENT (compact != true)"]];

Print["compact/true determinant ratio  (CONST => compact form = true d^n JK projection):"];
Print[StringRepeat["-", 70]];
Do[
  Print[cfg[[1]],
   "  jk=1: ", ratioConst[cfg[[2]], 1],
   " | jk=2: ", ratioConst[cfg[[2]], 2],
   " | jk=3: ", ratioConst[cfg[[2]], 3]],
  {cfg, configs}];
Print[StringRepeat["-", 70]];

(* also show the p(p-1) double-pole coefficient in the TRUE projection *)
Print[];
Print["True d^2 projection, coeff of double pole 1/(z1-z2)^2  (expect p(p-1)):"];
Do[
  Module[{orb = true[1, 2, 0, p], series},
   series = Series[orb*(z[1] - z[2])^2 /. z[1] -> z[2] + eps, {eps, 0, 0}];
   Print["  p=jk=", p, " : ", Simplify[Normal[series] /. eps -> 0],
     "   (p(p-1)=", p (p - 1), ")"]],
  {p, 1, 3}];
