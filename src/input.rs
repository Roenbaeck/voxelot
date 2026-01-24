use std::time::Instant;
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugView {
    None,
    Ssao,
    Ssr,
    Hzb,
    GiCombined,
    RadianceCascades,
}

impl DebugView {
    pub fn next(&self) -> Self {
        match self {
            DebugView::None => DebugView::Ssao,
            DebugView::Ssao => DebugView::Ssr,
            DebugView::Ssr => DebugView::Hzb,
            DebugView::Hzb => DebugView::GiCombined,
            DebugView::GiCombined => DebugView::RadianceCascades,
            DebugView::RadianceCascades => DebugView::None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DebugView::None => "None",
            DebugView::Ssao => "SSAO",
            DebugView::Ssr => "SSR",
            DebugView::Hzb => "HZB",
            DebugView::GiCombined => "GI Combined",
            DebugView::RadianceCascades => "Radiance Cascades",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurableSetting {
    FogDensity,
    BloomEnabled,
    SsaoEnabled,
    SsrEnabled,
    DofEnabled,
    KawaseEnabled,
    HzbEnabled,
    SsilvbSamples,
    SsaoRadius,
    LodDistance,
    DofFocalDistance,
    DofFocalRange,
    DofBlurStrength,
    KawaseIterations,
    KawaseOffset,
    WaterLevel,
}

impl ConfigurableSetting {
    pub fn next(&self) -> Self {
        use ConfigurableSetting::*;
        match self {
            FogDensity => BloomEnabled,
            BloomEnabled => SsaoEnabled,
            SsaoEnabled => SsrEnabled,
            SsrEnabled => DofEnabled,
            DofEnabled => KawaseEnabled,
            KawaseEnabled => HzbEnabled,
            HzbEnabled => SsilvbSamples,
            SsilvbSamples => SsaoRadius,
            SsaoRadius => LodDistance,
            LodDistance => DofFocalDistance,
            DofFocalDistance => DofFocalRange,
            DofFocalRange => DofBlurStrength,
            DofBlurStrength => KawaseIterations,
            KawaseIterations => KawaseOffset,
            KawaseOffset => WaterLevel,
            WaterLevel => FogDensity,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ConfigurableSetting::FogDensity => "Fog Density",
            ConfigurableSetting::BloomEnabled => "Bloom",
            ConfigurableSetting::SsaoEnabled => "SSAO",
            ConfigurableSetting::SsrEnabled => "SSR",
            ConfigurableSetting::DofEnabled => "DoF",
            ConfigurableSetting::KawaseEnabled => "Kawase",
            ConfigurableSetting::HzbEnabled => "HZB",
            ConfigurableSetting::SsilvbSamples => "SSILVB Samples",
            ConfigurableSetting::SsaoRadius => "SSAO Radius",
            ConfigurableSetting::LodDistance => "LOD Distance",
            ConfigurableSetting::DofFocalDistance => "DoF Distance",
            ConfigurableSetting::DofFocalRange => "DoF Range",
            ConfigurableSetting::DofBlurStrength => "DoF Strength",
            ConfigurableSetting::KawaseIterations => "Kawase Iters",
            ConfigurableSetting::KawaseOffset => "Kawase Offset",
            ConfigurableSetting::WaterLevel => "Water Level",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InputAction {
    ToggleTimePause,
    ToggleGui,
    ToggleVessel,
    ToggleFullscreen,
    SaveAndQuit,
    CycleDebugView,
    CycleSetting,
    SettingAdjust { direction: i32 },   // -1 or +1
    AdjustLodRender { direction: i32 }, // PageDown/Up
    AdjustFarPlane { direction: i32 },  // Z/C
    None,
}

pub struct InputManager {
    pub active_debug_view: DebugView,
    pub selected_setting: Option<ConfigurableSetting>,
    pub last_interaction: Instant,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            active_debug_view: DebugView::None,
            selected_setting: None,
            last_interaction: Instant::now(),
        }
    }

    pub fn map_key(&mut self, key: KeyCode) -> InputAction {
        match key {
            KeyCode::KeyT => InputAction::ToggleTimePause,
            KeyCode::F5 => InputAction::ToggleGui,
            KeyCode::KeyV => InputAction::ToggleVessel,
            KeyCode::F11 => InputAction::ToggleFullscreen,
            KeyCode::Escape => InputAction::SaveAndQuit,
            KeyCode::F3 => InputAction::CycleDebugView,
            KeyCode::Tab => InputAction::CycleSetting,
            KeyCode::Digit1 => InputAction::SettingAdjust { direction: -1 },
            KeyCode::Digit2 => InputAction::SettingAdjust { direction: 1 },
            KeyCode::PageDown => InputAction::AdjustLodRender { direction: -1 },
            KeyCode::PageUp => InputAction::AdjustLodRender { direction: 1 },
            KeyCode::KeyZ => InputAction::AdjustFarPlane { direction: -1 },
            KeyCode::KeyC => InputAction::AdjustFarPlane { direction: 1 },
            _ => InputAction::None,
        }
    }

    pub fn handle_action(&mut self, action: InputAction) {
        self.last_interaction = Instant::now();
        match action {
            InputAction::CycleDebugView => {
                self.active_debug_view = self.active_debug_view.next();
            }
            InputAction::CycleSetting => {
                if let Some(setting) = self.selected_setting {
                    self.selected_setting = Some(setting.next());
                } else {
                    self.selected_setting = Some(ConfigurableSetting::FogDensity);
                }
            }
            _ => {}
        }
    }
}
