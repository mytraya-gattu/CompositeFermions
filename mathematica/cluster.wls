#!/usr/bin/env wolframscript
(* JK-projected CF clustering test, N=4, disk.
   orbital_{n,m}(z_i) = z_i^(n+m) * e_n( { 1/(z_i-z_j) : j!=i }, each repeated jk times )
   psi = det[orbitals] * Prod_{i<j}(z_i-z_j)^(2 jk)
   Study leading power of eps as z1 -> z2 + eps, eps->0.  *)

Nn = 4;
others[i_] := Complement[Range[Nn], {i}];

(* e_n over the multiset where each 1/(z_i-z_j) appears jk times,
   via generating function Prod_{j!=i} (1 + t/(z_i-z_j))^jk, coeff of t^n *)
espMult[i_, n_, jk_] := Coefficient[
   Expand[Product[(1 + t/(z[i] - z[j]))^jk, {j, others[i]}]], t, n];

orbital[i_, n_, m_, jk_] := z[i]^(n + m) * espMult[i, n, jk];

(* Slater matrix: rows = particles i, cols = orbitals (n,m) in cfg *)
slater[cfg_, jk_] := Table[
   orbital[i, cfg[[a, 1]], cfg[[a, 2]], jk], {i, Nn}, {a, Length[cfg]}];

detPart[cfg_, jk_] := Det[slater[cfg, jk]];
jastrow[jk_] := Product[(z[i] - z[j])^(2 jk), {i, 1, Nn}, {j, i + 1, Nn}];

(* leading power of eps after z1 -> z2 + eps, with generic distinct numeric z2,z3,z4 *)
subsA = {z[2] -> 11/10, z[3] -> 13/7, z[4] -> 17/3};
subsB = {z[2] -> 3/5,   z[3] -> 23/4, z[4] -> -7/2};
leadExp1[f_, subs_] := Module[{ff, num, den},
   ff = Together[(f /. z[1] -> z[2] + eps) /. subs];
   num = Numerator[ff]; den = Denominator[ff];
   Exponent[num, eps, Min] - Exponent[den, eps, Min]];
(* report power; "*" appended if the two generic substitutions disagree *)
leadExp[f_] := Module[{a = leadExp1[f, subsA], b = leadExp1[f, subsB]},
   If[a === b, a, Row[{a, "/", b, "(!)"}]]];

configs = {
   {"C1: all LLL (nu_CF=1)",        {{0, 0}, {0, 1}, {0, 2}, {0, 3}}},
   {"C2: 3 in n=0, 1 in n=1",       {{0, 0}, {0, 1}, {0, 2}, {1, 0}}},
   {"C3: span n=0,1,2",             {{0, 0}, {0, 1}, {1, 0}, {2, 0}}},
   {"C4: 2 in n=0, 2 in n=1",       {{0, 0}, {0, 1}, {1, 0}, {1, 1}}}
   };

Print["jk = jk_type ;  detPow = leading eps power of det[orbitals] alone ;  psiPow = det * Jastrow^(2 jk)"];
Print["expected for C1 (Laughlin 1/(2jk+1)): psiPow = 2 jk + 1"];
Print[StringRepeat["-", 78]];
Do[
  Module[{name = cfg[[1]], c = cfg[[2]], dP, pP},
   Do[
    dP = leadExp[detPart[c, jk]];
    pP = leadExp[detPart[c, jk]*jastrow[jk]];
    Print[name, "  | jk=", jk, "  detPow=", dP, "  psiPow=", pP],
    {jk, 1, 3}];
   Print[StringRepeat["-", 78]];
   ],
  {cfg, configs}];
