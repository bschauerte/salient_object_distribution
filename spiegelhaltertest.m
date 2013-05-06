function pval = spiegelhaltertest(x)
  % SPIEGELHALTERTEST implements Spiegelhalter's test against a Gaussian
  %   distribution, see D. J. Spiegelhalter, "Diagnostic tests of 
  %   distributional shape," Biometrika, 1983
  %
  % Input:  x should be a vector
  % Output: p-value under null of x normally distributed
  % 
  % Original code taken from Stackoverflow (authored bz shabbychef)
  %
  % @author: B. Schauerte
  % @date:   2012

  xm = mean(x);
  xs = std(x);
  xz = (x - xm) ./ xs;
  xz2 = xz.^2;
  N = sum(xz2 .* log(xz2));
  n = numel(x);
  ts = (N - 0.73 * n) / (0.8969 * sqrt(n));
  pval = 1 - abs(erf(ts / sqrt(2)));