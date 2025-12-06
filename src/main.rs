use std::{array, fs};
mod words;
use genetica::{
    crossover::single_point_crossover,
    individual::{Generate, Individual, Mutate},
    population::{generate_population, sort_population_descending},
};
use serde::Deserialize;

use crate::words::{
    ADJECTIVES, ADJECTIVES_COUNT, ADVERBS, ADVERBS_COUNT, CONJUNCTIONS, CONJUNCTIONS_COUNT,
    DETERMINERS, DETERMINERS_COUNT, NOUNS, NOUNS_COUNT, PREPOSITIONS, PREPOSITIONS_COUNT, VERBS,
    VERBS_COUNT,
};

const PC: f32 = 0.6;
const PM: f32 = 0.05;

const WORD_COUNT: usize = 6;

const NOUN_RATE: f32 = 0.30;
const VERB_RATE: f32 = 0.20;
const ADVERB_RATE: f32 = 0.10;
const ADJECTIVE_RATE: f32 = 0.10;
const PREPOSITION_RATE: f32 = 0.10;
const DETERMINER_RATE: f32 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq)]
enum WordType {
    Noun,
    Verb,
    Adverb,
    Adjective,
    Conjunction,
    Preposition,
    Determiner,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GeneType<'a>(pub &'a str, pub WordType);

impl<'a> Generate for GeneType<'a> {
    fn generate() -> Self {
        let random_f32: f32 = rand::random_range(0.00..1.00);
        //This seems stupid
        let (word, word_type) = if random_f32 <= NOUN_RATE {
            (NOUNS[rand::random_range(0..NOUNS_COUNT)], WordType::Noun)
        } else if random_f32 <= (NOUN_RATE + VERB_RATE) {
            (VERBS[rand::random_range(0..VERBS_COUNT)], WordType::Verb)
        } else if random_f32 <= (NOUN_RATE + VERB_RATE + ADVERB_RATE) {
            (
                ADVERBS[rand::random_range(0..ADVERBS_COUNT)],
                WordType::Adverb,
            )
        } else if random_f32 <= (NOUN_RATE + VERB_RATE + ADVERB_RATE + ADJECTIVE_RATE) {
            (
                ADJECTIVES[rand::random_range(0..ADJECTIVES_COUNT)],
                WordType::Adjective,
            )
        } else if random_f32
            <= (NOUN_RATE + VERB_RATE + ADVERB_RATE + ADJECTIVE_RATE + PREPOSITION_RATE)
        {
            (
                PREPOSITIONS[rand::random_range(0..PREPOSITIONS_COUNT)],
                WordType::Preposition,
            )
        } else if random_f32
            <= (NOUN_RATE
                + VERB_RATE
                + ADVERB_RATE
                + ADJECTIVE_RATE
                + PREPOSITION_RATE
                + DETERMINER_RATE)
        {
            (
                DETERMINERS[rand::random_range(0..DETERMINERS_COUNT)],
                WordType::Determiner,
            )
        } else {
            (
                CONJUNCTIONS[rand::random_range(0..CONJUNCTIONS_COUNT)],
                WordType::Conjunction,
            )
        };
        GeneType(word, word_type)
    }
}

impl<'a> Mutate for GeneType<'a> {
    fn mutate(&mut self) {
        if rand::random_range(0.00..1.00) <= PM {
            let random_f32: f32 = rand::random_range(0.00..1.00);

            let (word, word_type) = if random_f32 <= NOUN_RATE {
                (NOUNS[rand::random_range(0..NOUNS_COUNT)], WordType::Noun)
            } else if random_f32 <= (NOUN_RATE + VERB_RATE) {
                (VERBS[rand::random_range(0..VERBS_COUNT)], WordType::Verb)
            } else if random_f32 <= (NOUN_RATE + VERB_RATE + ADVERB_RATE) {
                (
                    ADVERBS[rand::random_range(0..ADVERBS_COUNT)],
                    WordType::Adverb,
                )
            } else if random_f32 <= (NOUN_RATE + VERB_RATE + ADVERB_RATE + ADJECTIVE_RATE) {
                (
                    ADJECTIVES[rand::random_range(0..ADJECTIVES_COUNT)],
                    WordType::Adjective,
                )
            } else if random_f32
                <= (NOUN_RATE + VERB_RATE + ADVERB_RATE + ADJECTIVE_RATE + PREPOSITION_RATE)
            {
                (
                    PREPOSITIONS[rand::random_range(0..PREPOSITIONS_COUNT)],
                    WordType::Preposition,
                )
            } else if random_f32
                <= (NOUN_RATE
                    + VERB_RATE
                    + ADVERB_RATE
                    + ADJECTIVE_RATE
                    + PREPOSITION_RATE
                    + DETERMINER_RATE)
            {
                (
                    DETERMINERS[rand::random_range(0..DETERMINERS_COUNT)],
                    WordType::Determiner,
                )
            } else {
                (
                    CONJUNCTIONS[rand::random_range(0..CONJUNCTIONS_COUNT)],
                    WordType::Conjunction,
                )
            };

            self.0 = word;
            self.1 = word_type;
        };
    }
}

#[derive(Debug, Clone, Copy)]
struct Chromosome<'a> {
    genes: [GeneType<'a>; WORD_COUNT],
    fitness: Option<f32>,
}

impl<'a> Individual for Chromosome<'a> {
    type GeneType = GeneType<'a>;
    const GENES_SIZE: usize = WORD_COUNT;
    fn new() -> Self {
        let genes: [GeneType; WORD_COUNT] = array::from_fn(|_| GeneType::generate());
        Chromosome {
            genes,
            fitness: None,
        }
    }

    fn mutate_genes(&mut self) {
        for gene in &mut self.genes {
            gene.mutate();
        }
    }
    fn genes(&self) -> &[Self::GeneType] {
        &self.genes
    }
    fn genes_mut(&mut self) -> &mut [Self::GeneType] {
        &mut self.genes
    }

    fn fitness(&self) -> Option<f32> {
        self.fitness
    }
    fn fitness_mut(&mut self) -> &mut Option<f32> {
        &mut self.fitness
    }

    fn calculate_fitness(&mut self) {
        let mut fitness: f32 = 0.0;
        let mut previous_word: Option<&WordType> = None;
        let last_word = self.genes.last();

        self.genes.iter().for_each(|gt| {
            let word_type = &gt.1;
            if let Some(previous_word) = previous_word {
                match word_type {
                    WordType::Noun => match previous_word {
                        WordType::Noun => {
                            fitness -= 0.2;
                        }

                        WordType::Adjective => {
                            fitness += 0.2;
                        }
                        WordType::Conjunction => {
                            fitness -= 0.05;
                        }
                        WordType::Adverb => {
                            fitness -= 0.3;
                        }
                        WordType::Verb => {
                            fitness -= 0.1;
                        }
                        WordType::Preposition | WordType::Determiner => fitness += 0.3,
                    },
                    WordType::Verb => match previous_word {
                        WordType::Noun => {
                            if gt.0.ends_with("ing") {
                                fitness -= 0.1
                            } else if gt.0.ends_with("ed") || gt.0.ends_with("pt") {
                                fitness += 0.2
                            } else {
                                fitness += 0.1
                            }
                        }
                        WordType::Adjective => fitness -= 0.1,
                        _ => fitness -= 0.05,
                    },
                    WordType::Adverb => match previous_word {
                        WordType::Verb => {
                            fitness += 0.2;
                        }
                        WordType::Adverb => {
                            fitness -= 0.1;
                        }
                        _ => fitness -= 0.1,
                    },
                    WordType::Adjective => match previous_word {
                        WordType::Noun => fitness -= 0.2,
                        _ => fitness += 0.1,
                    },
                    WordType::Conjunction => match previous_word {
                        WordType::Noun => fitness += 0.1,
                        WordType::Conjunction => fitness -= 0.3,
                        _ => fitness -= 0.1,
                    },
                    WordType::Preposition => {}
                    WordType::Determiner => {}
                }
            } else {
                if word_type == &WordType::Noun {
                    fitness += 0.5;
                } else if word_type == &WordType::Adjective {
                    fitness += 0.3;
                } else {
                    fitness -= 0.1;
                }
            }
            previous_word = Some(word_type)
        });
        if let Some(last_word) = last_word {
            match last_word.1 {
                WordType::Conjunction => fitness -= 0.3,
                WordType::Adverb => fitness -= 0.1,
                _ => {}
            }
        };
        self.fitness = Some(fitness)
    }
}

#[derive(Deserialize)]
struct Config {
    pub generations: i32,
    pub population_count: i32,
}

fn main() {
    let config_data = fs::read_to_string("config.toml").unwrap();
    let config: Config = toml::from_str(&config_data).unwrap();

    let mut population: Vec<Chromosome> = generate_population(config.population_count);

    population.iter_mut().for_each(|c| c.calculate_fitness());

    for _ in 0..config.generations {
        sort_population_descending(&mut population);
        let parent1 = &population[0];
        let parent2 = &population[1];

        let (mut child1, mut child2) = single_point_crossover(parent1, parent2, PC);
        child1.mutate_genes();
        child2.mutate_genes();

        let mut new_population: Vec<Chromosome> = generate_population(config.population_count - 3);

        new_population.push(child1);
        new_population.push(child2);
        new_population.push(*parent1);

        new_population
            .iter_mut()
            .for_each(|c| c.calculate_fitness());
        population = new_population
    }

    sort_population_descending(&mut population);
    let best = &population[0];
    let best_constructed_word = best
        .genes
        .iter()
        .map(|g| g.0.to_string())
        .collect::<Vec<String>>()
        .join(" ");
    println!(
        "Fitness: {}\nWord: {}.",
        best.fitness.unwrap(),
        best_constructed_word
    );
}

/*
fn total(genes: &[GeneType; ARRAYSIZE], array: [i32; ARRAYSIZE]) -> i32 {
    array
        .iter()
        .zip(genes.iter())
        .filter(|&(_, gene)| gene.0)
        .map(|(value, _)| value)
        .sum()
}
*/
