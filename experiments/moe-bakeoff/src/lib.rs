use std::borrow::Cow;

use rune::{AppSpec, Color, Cx, El, Phase, Pixels, col, el, row, text};
use rune::color::hex;
use rune_kit::{Cell, Kit, SortDir, Table, TableColumn, TableState};
use serde::Deserialize;

const HEATMAP_WIDTH: f32 = 640.0;
const HEATMAP_HEIGHT: f32 = 420.0;

#[derive(Deserialize)]
struct Evidence {
    layer_count: usize,
    expert_count: usize,
    samples: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    step: u32,
    layer: usize,
    routed_tokens: u32,
    counts: Vec<u32>,
    diversity: f64,
    overloaded: Vec<usize>,
    dead: Vec<usize>,
}

impl Sample {
    fn share(&self, expert: usize) -> f64 {
        f64::from(self.counts[expert - 1]) / f64::from(self.routed_tokens)
    }
}

struct Expert {
    key: String,
    id: usize,
    share: f64,
    delta: f64,
    status: &'static str,
}

struct State {
    evidence: Evidence,
    incident: usize,
    selected: (usize, usize),
    hovered: Option<(usize, usize)>,
    heatmap: Pixels,
    experts: Vec<Expert>,
    table: TableState,
}

impl State {
    fn new() -> Self {
        let evidence: Evidence = serde_json::from_str(include_str!("../artifacts/stats.json"))
            .expect("the generated fixture must be valid");
        let incident = evidence
            .samples
            .iter()
            .position(|sample| !sample.overloaded.is_empty() || !sample.dead.is_empty())
            .expect("the fixture must contain an incident");
        let event = &evidence.samples[incident];
        let selected = (event.layer, event.overloaded.first().or(event.dead.first()).copied().unwrap());
        let heatmap = Pixels::new();
        let rgba = evidence
            .samples
            .iter()
            .filter(|sample| sample.step == event.step)
            .flat_map(|sample| {
                (1..=evidence.expert_count).flat_map(move |expert| {
                    if sample.counts[expert - 1] == 0 {
                        [236, 84, 99, 255]
                    } else if sample.overloaded.contains(&expert) {
                        [255, 190, 94, 255]
                    } else {
                        let intensity = (sample.share(expert) * 100.0).min(1.0);
                        [21 + (15.0 * intensity) as u8, 44 + (40.0 * intensity) as u8, 65 + (54.0 * intensity) as u8, 255]
                    }
                })
            })
            .collect();
        heatmap.set(evidence.expert_count as u32, evidence.layer_count as u32, rgba);

        let mut state = Self {
            evidence,
            incident,
            selected,
            hovered: None,
            heatmap,
            experts: Vec::new(),
            table: TableState::new(),
        };
        state.table.set_sort("delta", SortDir::Descending);
        state.select(selected);
        state
    }

    fn select(&mut self, cell: (usize, usize)) {
        self.selected = cell;
        let (layer, expert) = cell;
        let step = self.evidence.samples[self.incident].step;
        let sample = self.sample(step, layer);
        let baseline = self
            .evidence
            .samples
            .iter()
            .rev()
            .find(|sample| sample.layer == layer && sample.step < step)
            .expect("the fixture must contain a baseline for each layer");
        self.experts = (1..=self.evidence.expert_count)
            .map(|id| Expert {
                key: id.to_string(),
                id,
                share: sample.share(id),
                delta: sample.share(id) - baseline.share(id),
                status: if sample.dead.contains(&id) {
                    "Dead"
                } else if sample.overloaded.contains(&id) {
                    "Overloaded"
                } else {
                    "Active"
                },
            })
            .collect();
        if let Some((column, direction)) = self.table.sort() {
            self.sort(&column.to_owned(), direction);
        }
        self.table.select(expert.to_string());
    }

    fn sample(&self, step: u32, layer: usize) -> &Sample {
        self.evidence.samples.iter().find(|sample| sample.step == step && sample.layer == layer)
            .expect("the fixture must have each layer at each sampled step")
    }

    fn sort(&mut self, column: &str, direction: SortDir) {
        self.experts.sort_by(|left, right| {
            let order = match column {
                "expert" => left.id.cmp(&right.id),
                "share" => left.share.total_cmp(&right.share),
                "status" => left.status.cmp(right.status),
                _ => left.delta.total_cmp(&right.delta),
            };
            direction.apply(order).then_with(|| left.id.cmp(&right.id))
        });
    }

    fn cell_at(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        if !(0.0..HEATMAP_WIDTH).contains(&x) || !(0.0..HEATMAP_HEIGHT).contains(&y) {
            return None;
        }
        let layer = (y / HEATMAP_HEIGHT * self.evidence.layer_count as f32) as usize + 1;
        let expert = (x / HEATMAP_WIDTH * self.evidence.expert_count as f32) as usize + 1;
        Some((layer, expert))
    }
}

fn heatmap(state: &State, kit: &Kit) -> El {
    let (layer, expert) = state.selected;
    let cell_width = HEATMAP_WIDTH / state.evidence.expert_count as f32;
    let cell_height = HEATMAP_HEIGHT / state.evidence.layer_count as f32;
    let (focus_layer, focus_expert) = state.hovered.unwrap_or(state.selected);
    let incident = &state.evidence.samples[state.incident];
    let sample = state.sample(incident.step, focus_layer);
    let caption = format!(
        "Layer {focus_layer:02} / Expert {focus_expert:03}  ·  {:.2}% routing share",
        sample.share(focus_expert) * 100.0
    );

    col().gap(10.0)
        .child(text("Routing load by layer and expert").font(kit.ty.body_medium).color(kit.c.text_base))
        .child(el().key("routing-heatmap").size(HEATMAP_WIDTH, HEATMAP_HEIGHT)
            .label("Routing heatmap, 60 layers by 128 experts").pixels(&state.heatmap)
            .on_pointer_move_local(|state: &mut State, point, tx| {
                let hovered = state.cell_at(point.x, point.y);
                if hovered == state.hovered { tx.live_only(); }
                state.hovered = hovered;
            })
            .on_hover_change(|state: &mut State, entered, _| {
                if !entered { state.hovered = None; }
            })
            .on_drag(|state: &mut State, drag, _| {
                if drag.phase == Phase::Up && !drag.cancelled()
                    && let Some(cell) = state.cell_at(drag.x, drag.y)
                {
                    state.select(cell);
                }
            })
            .child(el().absolute((expert - 1) as f32 * cell_width, (layer - 1) as f32 * cell_height)
                .size(cell_width, cell_height).outline(1.0, 1.0, Color::WHITE)))
        .child(text(caption).key("cell-readout").font(kit.ty.text_small).color(kit.c.text_base))
        .child(row().gap(16.0)
            .child(legend("#214b6b", "Active", kit))
            .child(legend("#ffbe5e", "Overloaded", kit))
            .child(legend("#ec5463", "Dead", kit)))
}

fn legend(color: &str, label: &str, kit: &Kit) -> El {
    row().center_y().gap(6.0)
        .child(el().size(8.0, 8.0).fill(hex(color).expect("valid legend color")))
        .child(text(label).font(kit.ty.text_small).color(kit.c.text_muted))
}

fn expert_table(state: &State, kit: &Kit) -> El {
    Table::new(
        |s: &State| &s.table,
        |s: &mut State| &mut s.table,
        |s: &State| s.experts.len(),
        |s: &State, index| Cow::Borrowed(s.experts[index].key.as_str()),
    )
    .name("Routing by expert")
    .column(TableColumn::new("expert", "Expert").width(66.0).sortable())
    .column(TableColumn::new("share", "Load").width(68.0).sortable())
    .column(TableColumn::new("delta", "Δ load (pp)").width(118.0).sortable())
    .column(TableColumn::new("status", "Status").flex(1.0).min(90.0).sortable())
    .cell(|state: &State, index, column| {
        let expert = &state.experts[index];
        match column {
            "expert" => Cell::text(format!("#{:03}", expert.id)).mono(),
            "share" => Cell::text(format!("{:.2}%", expert.share * 100.0)).mono(),
            "delta" => Cell::text(format!("{:+.2}", expert.delta * 100.0)).mono(),
            _ => Cell::text(expert.status),
        }
    })
    .on_sort(|s: &mut State, column, direction, _| s.sort(column, direction))
    .on_select(|s: &mut State, key, _| {
        if let Ok(expert) = key.parse() { s.selected.1 = expert; }
    })
    .zebra(true)
    .build(state, kit)
    .h(420.0)
}

fn view(state: &State, cx: &mut Cx<State>) -> El {
    let kit = Kit::neutral(cx.system_dark());
    let incident = &state.evidence.samples[state.incident];
    let summary = format!("First sampled bad step {}  ·  Layer {:02}  ·  Diversity {:.2}",
        incident.step, incident.layer, incident.diversity);
    let dead = incident.dead.iter().map(|id| format!("{id:02}")).collect::<Vec<_>>().join(", ");
    let overloaded = incident.overloaded.iter().map(|id| {
        format!("Expert {id} ({:.0}%)", incident.share(*id) * 100.0)
    }).collect::<Vec<_>>().join(", ");
    let finding = format!("Overloaded: {overloaded}  ·  Dead experts: {dead}");

    col().w_full().h_full().fill(kit.c.page).padding(24.0).gap(20.0)
        .child(col().gap(6.0).padding(14.0).radius(8.0).fill(kit.c.bg_base)
            .child(text("MoE routing imbalance").font(rune::TextStyle::new(23.0, 650.0)).color(kit.c.text_base))
            .child(text(summary).key("incident-summary").font(kit.ty.body_medium).color(kit.c.text_base))
            .child(text(finding).key("incident-experts").font(kit.ty.text_small).color(kit.c.text_muted)))
        .child(row().w_full().gap(24.0)
            .child(heatmap(state, &kit))
            .child(col().grow(1.0).min_w(350.0).gap(10.0)
                .child(text(format!("Layer {:02} expert load", state.selected.0))
                    .key("selected-layer").font(kit.ty.body_medium).color(kit.c.text_base))
                .child(expert_table(state, &kit))))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen(start))]
pub fn start() {
    rune::run(AppSpec::new("TensorQuill · MoE inspector", (1160.0, 720.0), State::new, view));
}
