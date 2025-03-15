use rand::Rng;

#[derive(Default, Clone)]
pub struct Biome {
    pub name: String,
    pub kind: BiomeKind,
}

#[derive(Clone)]
pub enum BiomeKind {
    Desert,
    Tundra,
    Jungle,
    Grasslands,
}

impl BiomeKind {
    pub fn name(&self) -> &str {
        match self {
            BiomeKind::Desert => "Desert",
            BiomeKind::Grasslands => "Grasslands",
            BiomeKind::Tundra => "Tundra",
            BiomeKind::Jungle => "Jungle",
        }
    }

    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let val = rng.gen_range(0_u32..4);
        println!("RAND IS {val}");
        match val {
            0 => BiomeKind::Desert,
            1 => BiomeKind::Tundra,
            2 => BiomeKind::Jungle,
            _ => BiomeKind::Grasslands,
        }
    }

    pub fn color(&self) -> [f32; 3] {
        match self {
            BiomeKind::Desert => [0.46, 0.33, 0.25],
            BiomeKind::Grasslands => [0.38, 0.49, 0.20],
            BiomeKind::Jungle => [0.14, 0.45, 0.11],
            BiomeKind::Tundra => [0.4, 0.4, 0.37],
        }
    }
}

impl Default for BiomeKind {
    fn default() -> Self {
        Self::Grasslands
    }
}

impl Biome {
    pub fn random() -> Self {
        let mut biome = Biome::default();
        biome.kind = BiomeKind::random();

        biome.name = biome.kind.name().into();

        biome
    }
}
