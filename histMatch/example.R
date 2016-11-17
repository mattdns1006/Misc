library(ggplot2)
n = 1e6
src = rnorm(n,1,1)
tar = rweibull(n,2,10)

cdf <- function(x){
  binned = cut(x,200)
  cs = cumsum(table(binned))
  cs = cs/length(x)
  return(cs)  
}

getClosest <- function(vec,num){
  return(which(abs(vec-num)==min(abs(vec-num)))  )
}

srcCdf = cdf(src)
tarCdf = cdf(tar)

df = data.frame(
  interval = seq(1,250,length.out = length(srcCdf)),
  Source = srcCdf,
  Target = tarCdf
)

x0 = 125
fx0 = df$Source[getClosest(df$interval,x0)]
fx1 = df$Target[getClosest(df$Target,fx0)]
x1 = df$interval[getClosest(df$Target,fx0)]
  
arrowThick = 0.3
ggplot(df,aes(interval)) + 
  geom_line(aes(y = Source),col="red") + 
  geom_line(aes(y = Target),col="blue") + 
  xlab("Pixel intensity") +
  ylab("Percentage of total mass") +
  geom_segment(aes(x = x0, y = 0, xend = x0, yend = fx0), arrow = arrow(length = unit(arrowThick, "cm"))) + 
  geom_segment(aes(x = x0, y = fx0, xend = x1, yend = fx0), arrow = arrow(length = unit(arrowThick, "cm"))) +
  geom_segment(aes(x = x1, y = fx0, xend = x1, yend = 0), arrow = arrow(length = unit(arrowThick, "cm")))
  
ggsave(filename = "~/University/writeUps/15 Month Report/images/whale/histMatch/histMatch.jpg")



