import numpy as np
from random import Random

def fitness(population, n):
  
    #index_fit = 0
    population_fit = []
    flag = False
    for i in range(len(population)):
      chromosome = []
      if (len(population) != n):
        chromosome.append(population[i])
      else:
        chromosome = population
        flag = True

      horizontal_clash = abs(len(chromosome) - len(np.unique(chromosome)))
      diagonal_clashes = 0
      # calculate diagonal clashes
      for i in range(len(chromosome)):
          for j in range(i,len(chromosome)):
              if ( i != j):
                  dx = abs(i-j)
                  dy = abs(chromosome[i] - chromosome[j])
                  if(dx == dy):
                      diagonal_clashes += 1
        
      fit = 28 - (horizontal_clash + diagonal_clashes)
      population_fit.append(fit)
      if (flag):
        return population_fit
      #index_fit += 1  
    return population_fit


def select(population, fit):

    temp = population
    a = []
    for i in range(len(temp)):
      a.append(i)
    size = 1
    fit_lst = fit
    p = []
    for i in range(len(fit_lst)):
      t2 = 0.0
      t1 = float(fit_lst[i])
      for j in range(len(fit_lst)):
        t2 += fit_lst[j]
      p.append(t1/t2)

    #p = [.31, .29, 0.26, 0.14]

    index_num = np.random.choice(a, size, True, p)
    rand_select = temp[int(index_num)]
    return rand_select


def crossover(x, y):

    r = Random()
    idx = r.randint(3,5)
    x = x[0:idx]
    y = y[idx:8]
    #child = x + y
    #child = x[0:idx] + y[idx:8]
    child = np.concatenate((x , y))
    return child
    #return x[0:idx] + y[idx:8]

def mutate(child):

    r = Random()
    idx = r.randint(0, 7) #random index
    child[idx] = r.randint(1,8) #replaces selected random index with a random integer

    return child 

def GA(population, n, mutation_threshold = 0.3):

    gen_max = 100000
    gen_counter = gen_max
    target_fitness = 28
    while gen_counter > 0:
      new_population = []
      for i in range(len(population)):
        x = select(population, fitness(population, n))
        y = select(population, fitness(population, n))
        child = crossover(x, y)
        r = Random()
        if r.uniform(0, 1) < mutation_threshold:
          child = mutate(child)
        fit_child = fitness(child,n)
        if (fit_child[0] == target_fitness):
          print(child, " found in ", gen_max-gen_counter," generations\n")
          print("The max fitness value = ", fit_child)
          return child
        new_population.append(child)
      population = new_population
      gen_counter -= 1
    print("nothing found in ", gen_max, " genrations\n")
    return None

'''for 8 queen problem, n = 8'''
n = 8

'''start_population denotes how many individuals/chromosomes are there
  in the initial population n = 8'''
start_population = 10 

'''if you want you can set mutation_threshold to a higher value,
   to increase the chances of mutation'''
mutation_threshold = 0.3

'''creating the population with random integers between 0 to 7 inclusive
   for n = 8 queen problem'''
population = np.random.randint(0, n, (start_population, n))

'''calling the GA function'''
GA(population, n, mutation_threshold)

