def hines_solver(d, u, b, p):
  """
  d : diagonal of the matrix to be solved
  u : upper part with indices 0 .. N
  """
  N = d.shape[0]

  for i in range(N-1,0,-1):
    f = u[p[i]] / d[i]
    d = d.at[p[i]].set(d[p[i]] - f * u[p[i]])
    b = b.at[p[i]].set(b[p[i]] - f * b[i])

  b = b.at[0].set(b[0] / d[0])

  for i in range(1,N):
    b = b.at[i].set((b[i] - u[p[i]] * b[p[i]]) / d[i])

  return b