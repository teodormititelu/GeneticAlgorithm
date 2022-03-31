import random
import math
import numpy

f = open('Input.txt', 'r')
g = open('Evolutie.txt', 'w')

def convert_to_value( chromosome ):
    val = int("".join(str(bit) for bit in chromosome), 2)
    val = round( ((b - a) * val / (2 ** chromosomeLength - 1)) + a, precision )
    return  val

def func( value ):
    return x * (value * value) + y * value + z

def compute( population ):
    values = [convert_to_value(chromosome) for chromosome in population]
    fitnesses = [func(val) for val in values]
    return values, fitnesses


def print_population( population ):

    values, fitnesses = compute( population )

    maxLenVal = max( [ len(str(val)) for val in values ] )
    generationLen = len(str(populationSize))
    formatString = '{:>' + str(generationLen) + '} {} {} {} {:>' + str(maxLenVal) + '} {} {}'

    for i in range( populationSize ):
        g.write( formatString.format( i+1, ":", "".join(str(bit) for bit in population[i] ), "x=", '{:6f}'.format(values[i]), "f=", fitnesses[i] ) + '\n')

def upper_bound( arr, lf, rg, val ):
    if lf > rg:
        return populationSize
    mid = (lf + rg) // 2

    if (arr[mid] < val):
        return upper_bound( arr, mid+1, rg, val)
    return min( mid, upper_bound(arr, lf, mid-1, val))

def selection( generation ):
    if generation == 1:
        g.write("\nProbabilitati selectie:\n")

    generationLen = len(str(populationSize))
    formatString = '{:>' + str(generationLen) + '}'

    values,fitnesses = compute( population )

    fitSum = sum( fitnesses )
    if generation == 1:
        for i in range(populationSize):
            g.write( 'cromozom ' + formatString.format( i+1 ) + ' probabilitate ' + str(fitnesses[i]/fitSum) + '\n')

    if generation == 1:
        g.write("\nIntervale probabilitati selectie\n")

    probSelIntervals = [ fitnesses[0]/fitSum ]
    for i in range( 1,populationSize ):
        probSelIntervals.append( probSelIntervals[i-1] + fitnesses[i]/fitSum )
    if generation == 1:
        g.write( ' '.join( map(str, probSelIntervals) ) + '\n\n')

    newPopulation = []
    randList = numpy.random.uniform( 0, 1, populationSize )
    for i in range(populationSize):
        u = randList[i]
        chrIndex = upper_bound( probSelIntervals, 0, populationSize-1, u)
        newPopulation.append( population[chrIndex] )
        if generation == 1:
            g.write("u = " + str(u) + " selectez cromozomul " + str(chrIndex + 1) + '\n')

    return newPopulation

def crossover( population, generation = None):

    if generation == 1:
        g.write('\nProbabilitate de recombinare pentru fiecare gena ' + str(crossoverRate) + '\n')

    crossoverList = []
    
    randList = numpy.random.uniform( 0, 1, populationSize )
    for i in range(populationSize):
        u = randList[i]
        if u <= crossoverRate:
            crossoverList.append( (population[i], i) )

    if len( crossoverList ) % 2 == 1:
        crossoverList.pop()

    crossoverListLength = len( crossoverList )
    for i in range(0, crossoverListLength, 2):
        firstIndex = crossoverList[i][1]
        firstChr = crossoverList[i][0]
        secondIndex = crossoverList[i+1][1]
        secondChr = crossoverList[i+1][0]

        u = numpy.random.randint(0, chromosomeLength)
        newChr1 = firstChr[:u] + secondChr[u:]
        newChr2 = secondChr[:u] + secondChr[u:]

        population[firstIndex], population[secondIndex] = newChr1, newChr2

        if generation == 1:
            g.write( 'Recombinare dintre cromozomul ' + str(firstIndex+1) + ' cu cromozomul ' + str(secondIndex+1) + ':\n' )
            g.write( ''.join( map( str, firstChr) ) + ' ' + ''.join(map( str, secondChr)) + ' punct ' + str(u) + '\n' )
            g.write( 'Rezultat ' + ''.join(map( str, newChr1) ) + ' ' + ''.join(map( str, newChr2) ) + '\n')

    return population

def mutation( population, generation = None):

    modified = []

    randList = numpy.random.uniform( 0, 1, populationSize*chromosomeLength )
    for i in range( populationSize ):
        for j in range( chromosomeLength ):
            u = randList[i*j]
            if u <= mutationRate:
                population[i][j] = population[i][j] ^ 1
                modified.append( i+1 )

    modified = list(dict.fromkeys(modified))

    if generation == 1:
        g.write( '\nProbabilitate de mutatie pentru fiecare gena ' + str(mutationRate) + '\n')
        g.write( 'Au fost modificati cromozomii:\n')
        for index in modified:
            g.write( str(index) + '\n')

    return population

populationSize = int( f.readline() )
a,b = list( map( float, f.readline().split() ) )
x, y, z = list( map( float, f.readline().split() ) )
precision = int( f.readline() )
crossoverRate = float( f.readline() )
mutationRate = float( f.readline() )
generations = int( f.readline() )

values = []
fitnesses = []

chromosomeLength = math.ceil(math.log2((b-a)*(10**precision)))

population = [[ random.randint(0, 1) for bit in range(chromosomeLength)  ] for i in range(populationSize)]

for generation in range( 1, generations+1 ):
    if generation == 1:
        g.write("\nPopulatia initiala:\n")
        print_population( population )

    population = selection( generation )
    if generation == 1:
        g.write('\nDupa selectie:\n')
        print_population( population )

    population = crossover( population, generation )
    if generation == 1:
        g.write('\nDupa recombinare:\n')
        print_population( population )

    population = mutation( population, generation )
    if generation == 1:
        g.write('\nDupa mutatie:\n')
        print_population( population )

    values, fitnesses = compute( population )

    if generation == 1:
        g.write( '\nEvolutia maximului\n' )

    g.write( str(max( fitnesses )) + '\n')