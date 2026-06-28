#!/usr/bin/env wolframscript
(* Fractional jk_type: e_n via generalized-binomial generating function
   Prod_j (1 + t/(z_i-z_j))^jk  (analytic continuation of the multiset ESP).
   Question: does fractional jk give an acceptable (vanishing) wavefunction? *)

Nn = 4;
others[i_] := Complement[Range[Nn], {i}];

(* generalized e_n: works for any (real) jk via series in t *)
espFrac[i_, n_, jk_] := Coefficient[
   Normal[Series[Product[(1 + t/(z[i] - z[j]))^jk, {j, others[i]}], {t, 0, n}]],
   t, n];
orb[i_, n_, m_, jk_] := z[i]^(n + m)*espFrac[i, n, jk];
detOf[cfg_, jk_] := Det[Table[orb[i, cfg[[a, 1]], cfg[[a, 2]], jk], {i, Nn}, {a, Length[cfg]}]];

subs = {z[2] -> 11/10, z[3] -> 13/7, z[4] -> 17/3};
detPole[cfg_, jk_] := Module[{ff, num, den},
   ff = Together[(detOf[cfg, jk] /. z[1] -> z[2] + eps) /. subs];
   num = Numerator[ff]; den = Denominator[ff];
   Exponent[num, eps, Min] - Exponent[den, eps, Min]];

(* show the n=2 single-orbital double-pole coefficient as a function of jk *)
e2 = espFrac[1, 2, p];
dbl = Simplify[Coefficient[Together[e2], 1/(z[1] - z[2]), 2]];
Print["coeff of 1/(z1-z2)^2 in e_2 (symbolic jk=p): ", dbl, "   [= p(p-1)/2]"];
Print[StringRepeat["-", 64]];

cfgC3 = {{0, 0}, {0, 1}, {1, 0}, {2, 0}};  (* spans n=0,1,2 *)
Print["C3 (span n=0,1,2):  detPole  | psiPow = detPole + 2 jk"];
Do[
  Module[{dp = detPole[cfgC3, jk]},
   Print["  jk=", jk, "  detPole=", dp,
     "   psiPow=", dp + 2 jk,
     If[NumberQ[dp + 2 jk] && dp + 2 jk <= 0, "   <-- DIVERGES / non-vanishing!", ""]]],
  {jk, {1/2, 1, 6/5, 3/2, 2, 5/2, 3}}];
