n = 1e6
src = rnorm(n,1,1)
tar = rgamma(n,3,1)

cdf <- function(x){
  binned = cut(x,200)
  cs = cumsum(table(binned))
  return(cs)  
}

srcCdf = cdf(src)
tarCdf = cdf(tar)

