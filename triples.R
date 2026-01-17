B=1e5

n.sum20=0

for (b in 1:B) {
  trip=sample(x=1:18, size=3, replace=TRUE)
  n.sum20=n.sum20+(sum(trip)==20)
}

cat("n.sum20 = ", n.sum20, "\n")
A1.est=(n.sum20/B) * 18^3
cat("|A1| est = ", A1.est, "\n")
