/**
 * How each part of the anatomy is shaded.
 *
 * Pure data and no three.js, for the same reason the geometry is: the renderer
 * applies it, and a test can read it. Before this table the whole cell was
 * shaded by one string test — "areas that start with `terminal`, or the can, are
 * metallic; everything else is plastic" — which is why nickel-plated steel, a
 * polyethylene membrane and a pool of electrolyte all came out looking alike.
 *
 * These are the right *kinds* of value rather than measured ones: metals get a
 * high metalness and a roughness that says polished or machined, dielectrics get
 * zero metalness, and the environment contribution is nudged per part so the
 * reflective ones read as reflective. Nothing here is a data claim, and nothing
 * here changes a colour or a size — those come from the spec.
 */
export interface PartMaterial {
  metalness: number;
  roughness: number;
  /** How strongly the scene's environment map shows in this surface. */
  envMapIntensity: number;
}

/**
 * Where a material is not a bulk material, the *surface the viewer sees* wins:
 * the anode ribbon is copper foil, but what faces the camera is its graphite
 * coating, so it is shaded as graphite — matte and barely metallic — and the
 * folder is named in the comment rather than drawn as another layer.
 */
export const PART_MATERIALS: Record<string, PartMaterial> = {
  /** Nickel-plated stainless can. */
  can: { metalness: 0.85, roughness: 0.34, envMapIntensity: 1.0 },
  /** The top cap assembly, same steel, a coarser finish. */
  cap: { metalness: 0.85, roughness: 0.42, envMapIntensity: 0.9 },
  /** Machined aluminium burst disc. */
  vent: { metalness: 0.8, roughness: 0.35, envMapIntensity: 0.9 },
  /** The button and base, polished where a contact lands. */
  terminal_pos: { metalness: 0.92, roughness: 0.18, envMapIntensity: 1.15 },
  terminal_neg: { metalness: 0.92, roughness: 0.22, envMapIntensity: 1.1 },
  /** Aluminium cathode tab. */
  tab_pos: { metalness: 0.75, roughness: 0.3, envMapIntensity: 0.9 },
  /** Copper anode tab. */
  tab_neg: { metalness: 0.9, roughness: 0.28, envMapIntensity: 1.0 },
  /** Aluminium foil under a cathode coating. */
  cathode_sheet: { metalness: 0.5, roughness: 0.45, envMapIntensity: 0.7 },
  /** Copper foil under a graphite anode coating — the coating is what shows. */
  anode_sheet: { metalness: 0.25, roughness: 0.7, envMapIntensity: 0.4 },
  /** Polyethylene membrane: a dielectric, and matte about it. */
  separator: { metalness: 0.0, roughness: 0.85, envMapIntensity: 0.35 },
  /** Liquid electrolyte: a wet surface and very few rough features. */
  electrolyte: { metalness: 0.0, roughness: 0.12, envMapIntensity: 0.9 },
  /** Graphite powder, drawn as a legibility-limited sample. */
  particles: { metalness: 0.2, roughness: 0.8, envMapIntensity: 0.35 },
  /** The SEI: a thin dielectric film, slightly glossy. */
  sei_film: { metalness: 0.0, roughness: 0.55, envMapIntensity: 0.6 },
};

/** What a part with no entry gets: a neutral dielectric, clearly not metal. */
export const DEFAULT_PART_MATERIAL: PartMaterial = {
  metalness: 0.1,
  roughness: 0.6,
  envMapIntensity: 0.6,
};

export function materialFor(partId: string): PartMaterial {
  return PART_MATERIALS[partId] ?? DEFAULT_PART_MATERIAL;
}
