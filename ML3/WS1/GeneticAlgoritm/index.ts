import { getData } from "./data";
import { DataPoint, Individual } from "./types";

// Functie om een willekeurige getal binnen een bepaald bereik te genereren
function randomInRange(min: number, max: number): number {
	return Math.random() * (max - min) + min;
}

// Functie om een willekeurig individu te genereren
function generateRandomIndividual(): Individual {
	const individual: Individual = {
		a: randomInRange(-10, 10),
		b: randomInRange(-10, 10),
		c: randomInRange(-10, 10),
		d: randomInRange(-10, 10),
		e: randomInRange(-10, 10),
		f: randomInRange(-10, 10),
	};
	return individual;
}

// Functie om de fitnesswaarde van een individu te berekenen
function calculateFitness(
	individual: Individual,
	dataset: DataPoint[]
): number {
	let fitness = 0;
	for (let i = 0; i < dataset.length; i++) {
		const x = dataset[i].x;
		const y = dataset[i].y;
		const predictedY =
			individual.a * Math.pow(x, 5) +
			individual.b * Math.pow(x, 4) +
			individual.c * Math.pow(x, 3) +
			individual.d * Math.pow(x, 2) +
			individual.e * x +
			individual.f;
		fitness += Math.pow(y - predictedY, 2); // Quadratic error
	}
	return fitness;
}

// Functie om de beste individuen te selecteren op basis van hun fitnesswaarden
function selectBestIndividuals(
	population: Individual[],
	dataset: DataPoint[],
	k: number
): Individual[] {
	const sortedPopulation = [...population].sort((a, b) =>
		calculateFitness(a, dataset) < calculateFitness(b, dataset) ? -1 : 1
	);
	return sortedPopulation.slice(0, k);
}

// Functie voor eenpuntskruising
function crossover(parent1: Individual, parent2: Individual): Individual {
	// Determine the crossover point randomly between 0 and 5
	const crossoverPoint = Math.floor(Math.random() * 6);

	// Initialize the child with default values
	const child: Individual = {
		a: 0,
		b: 0,
		c: 0,
		d: 0,
		e: 0,
		f: 0,
	};

	// Copy properties from parent1 up to the crossover point
	for (let i = 0; i < crossoverPoint; i++) {
		const propertyKey = String.fromCharCode(97 + i);
		child[propertyKey] = parent1[propertyKey];
	}

	// Copy properties from parent2 from the crossover point onwards
	for (let i = crossoverPoint; i < 6; i++) {
		const propertyKey = String.fromCharCode(97 + i);
		child[propertyKey] = parent2[propertyKey];
	}

	return child;
}

// Functie voor mutatie
function mutate(individual: Individual, mutationRate: number): Individual {
	const mutatedIndividual: Individual = { ...individual };
	for (let i = 0; i < 6; i++) {
		if (Math.random() < mutationRate) {
			mutatedIndividual[String.fromCharCode(97 + i)] = randomInRange(
				-10,
				10
			);
		}
	}
	return mutatedIndividual;
}

// Hoofdgenetisch algoritme
function geneticAlgorithm(
	dataset: DataPoint[],
	populationSize: number,
	generations: number,
	mutationRate: number,
	k: number
): Individual {
	let population: Individual[] = [];
	for (let i = 0; i < populationSize; i++) {
		population.push(generateRandomIndividual());
	}

	for (let generation = 0; generation < generations; generation++) {
		const bestIndividuals = selectBestIndividuals(population, dataset, k);
		console.log(
			`Generation ${generation}: Best Fitness = ${calculateFitness(
				bestIndividuals[0],
				dataset
			)}`
		);

		// Genereer een nieuwe generatie
		const newPopulation: Individual[] = [];
		for (let i = 0; i < populationSize; i++) {
			const parent1 = bestIndividuals[Math.floor(Math.random() * k)];
			const parent2 = bestIndividuals[Math.floor(Math.random() * k)];
			const child = crossover(parent1, parent2);
			const mutatedChild = mutate(child, mutationRate);
			newPopulation.push(mutatedChild);
		}
		population = newPopulation;
	}

	return selectBestIndividuals(population, dataset, 1)[0];
}

const exampleDataset: DataPoint[] = getData();

const bestFit: Individual = geneticAlgorithm(exampleDataset, 100, 1, 0.1, 10);
console.log("Best Fit Parameters:", bestFit);
