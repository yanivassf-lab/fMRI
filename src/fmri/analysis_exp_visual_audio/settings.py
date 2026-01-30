import fnmatch




# =====================
# Brain Regions Mapping: Video vs. Music (Schaefer Atlas Based)
# =====================

regions_class_chatgpt = {
    # ===== VISUAL NETWORK =====
    "7Networks_*_Vis_*": {
        "dominant": "video",
        "video_weight": 0.95,
        "music_weight": 0.05,
        "confidence": 0.95
    },

    # ===== SOMATOMOTOR =====
    "7Networks_*_SomMot_*": {
        "dominant": "music",
        "video_weight": 0.65,
        "music_weight": 0.75,
        "confidence": 0.80
    },

    # ===== DORSAL ATTENTION =====
    "7Networks_*_DorsAttn_*": {
        "dominant": "video",
        "video_weight": 0.80,
        "music_weight": 0.60,
        "confidence": 0.75
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    "7Networks_*_SalVentAttn_*": {
        "dominant": "music",
        "video_weight": 0.75,
        "music_weight": 0.85,
        "confidence": 0.85
    },

    # ===== LIMBIC =====
    "7Networks_*_Limbic_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.90,
        "confidence": 0.90
    },

    # ===== CONTROL NETWORK =====
    "7Networks_*_Cont_*": {
        "dominant": "video",
        "video_weight": 0.70,
        "music_weight": 0.65,
        "confidence": 0.70
    },

    # ===== DEFAULT MODE – TEMPORAL =====
    "7Networks_*_Default_Temp_*": {
        "dominant": "music",
        "video_weight": 0.20,
        "music_weight": 0.65,
        "confidence": 0.65
    },

    # ===== DEFAULT MODE – PARIETAL =====
    "7Networks_*_Default_Par_*": {
        "dominant": "music",
        "video_weight": 0.15,
        "music_weight": 0.35,
        "confidence": 0.45
    },

    # ===== DEFAULT MODE – PFC =====
    "7Networks_*_Default_PFC_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.60,
        "confidence": 0.65
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "music",
        "video_weight": 0.25,
        "music_weight": 0.65,
        "confidence": 0.70
    },
}

regions_class_gemini = {
    # =================================================================
    # Brain Regions Mapping: Task-Specific Weights (Schaefer Atlas)
    # Based on: Zatorre (2007), Salimpoor (2013), and Fox et al. (2005)
    # =================================================================
    # ===== VISUAL NETWORK – Primary & Extrastriate =====
    "7Networks_*_Vis_*": {
        "dominant": "video",
        "video_weight": 0.95,
        "music_weight": 0.05,
        "confidence": 0.95,
        "notes": "Purely visual processing. Music weight is minimal (cross-modal inhibition)."
    },

    # ===== SOMATOMOTOR – Auditory Core & Rhythm =====
    # Note: Schaefer includes Heschl's Gyrus within SomMot
    "7Networks_*_SomMot_*": {
        "dominant": "music",
        "video_weight": 0.15,
        "music_weight": 0.85,
        "confidence": 0.90,
        "notes": "Captures the auditory cortex. Video weight accounts for action-observation/motor mirror."
    },

    # ===== DORSAL ATTENTION – Spatial & Tracking =====
    "7Networks_*_DorsAttn_*": {
        "dominant": "video",
        "video_weight": 0.80,
        "music_weight": 0.40,
        "confidence": 0.85,
        "notes": "Video triggers spatial tracking; Music triggers rhythmic/tonal attention."
    },

    # ===== SALIENCE / VENTRAL ATTENTION – Novelty Detection =====
    "7Networks_*_SalVentAttn_*": {
        "dominant": "both",
        "video_weight": 0.60,
        "music_weight": 0.55,
        "confidence": 0.75,
        "notes": "Active for scene cuts in video and melodic/rhythmic shifts in music."
    },

    # ===== LIMBIC – Emotion & Memory Poles =====
    "7Networks_*_Limbic_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.70,
        "confidence": 0.70,
        "notes": "Deeply linked to musical pleasure and reward (Striatal-Limbic loop)."
    },

    # ===== CONTROL NETWORK – Executive Functions =====
    "7Networks_*_Cont_*": {
        "dominant": "both",
        "video_weight": 0.65,
        "music_weight": 0.60,
        "confidence": 0.70,
        "notes": "Cognitive demand for narrative tracking in video or structural complexity in music."
    },

    # ===== DEFAULT MODE – Medial PFC (The "Convergence Zone") =====
    "7Networks_*_Default_PFC_*": {
        "dominant": "music",
        "video_weight": 0.20,
        "music_weight": 0.75,
        "confidence": 0.80,
        "notes": "Tracks musical tonality and self-referential emotional response."
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "music",
        "video_weight": 0.25,
        "music_weight": 0.65,
        "confidence": 0.70,
        "notes": "Nostalgia and autobiographical memory hub, highly resonant with music."
    },

    # ===== DEFAULT MODE – Temporal / Parietal nodes =====
    "7Networks_*_Default_Temp_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.80,
        "confidence": 0.85,
        "notes": "Integration of social and semantic auditory information."
    }
}




# ==========================================
#chatgpt 5.2
regions_class_chatgpt52 = {
    # ===== VISUAL NETWORK =====
    "7Networks_*_Vis_*": {
        "dominant": "video",
        "video_weight": 0.95,
        "music_weight": 0.05,
        "confidence": 0.95
    },

    # ===== DORSAL ATTENTION (visual attention, eye movements) =====
    "7Networks_*_DorsAttn_*": {
        "dominant": "video",
        "video_weight": 0.80,
        "music_weight": 0.20,
        "confidence": 0.80
    },

    # ===== SOMATOMOTOR (rhythm, beat, movement) =====
    "7Networks_*_SomMot_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.60,
        "confidence": 0.60
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    "7Networks_*_SalVentAttn_*": {
        "dominant": "music",
        "video_weight": 0.40,
        "music_weight": 0.65,
        "confidence": 0.70
    },

    # ===== LIMBIC (emotion, reward) =====
    "7Networks_*_Limbic_*": {
        "dominant": "music",
        "video_weight": 0.25,
        "music_weight": 0.80,
        "confidence": 0.85
    },

    # ===== CONTROL / FRONTOPARIETAL =====
    "7Networks_*_Cont_*": {
        "dominant": "music",
        "video_weight": 0.45,
        "music_weight": 0.55,
        "confidence": 0.55
    },

    # ===== DEFAULT MODE – PFC =====
    "7Networks_*_Default_PFC_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.60,
        "confidence": 0.65
    },

    # ===== DEFAULT MODE – TEMPORAL =====
    "7Networks_*_Default_Temp_*": {
        "dominant": "music",
        "video_weight": 0.30,
        "music_weight": 0.70,
        "confidence": 0.75
    },

    # ===== DEFAULT MODE – PARIETAL =====
    "7Networks_*_Default_Par_*": {
        "dominant": "music",
        "video_weight": 0.35,
        "music_weight": 0.65,
        "confidence": 0.70
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "music",
        "video_weight": 0.25,
        "music_weight": 0.65,
        "confidence": 0.70
    }
}

regions_class_claude_opus45 = {
    # ===== VISUAL NETWORK =====
    "7Networks_*_Vis_*": {
        "dominant": "video",
        "video_weight": 0.95,
        "music_weight": 0.05,
        "confidence": 0.95
    },

    # ===== DORSAL ATTENTION =====
    "7Networks_*_DorsAttn_*": {
        "dominant": "video",
        "video_weight": 0.75,
        "music_weight": 0.25,
        "confidence": 0.75
    },

    # ===== SOMATOMOTOR =====
    "7Networks_*_SomMot_*": {
        "dominant": "music",
        "video_weight": 0.35,
        "music_weight": 0.55,
        "confidence": 0.60
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    "7Networks_*_SalVentAttn_*": {
        "dominant": "music",
        "video_weight": 0.40,
        "music_weight": 0.60,
        "confidence": 0.65
    },

    # ===== LIMBIC =====
    "7Networks_*_Limbic_*": {
        "dominant": "music",
        "video_weight": 0.35,
        "music_weight": 0.65,
        "confidence": 0.65
    },

    # ===== CONTROL / FRONTOPARIETAL =====
    "7Networks_*_Cont_*": {
        "dominant": "balanced",
        "video_weight": 0.50,
        "music_weight": 0.50,
        "confidence": 0.70
    },

    # ===== DEFAULT MODE – PFC =====
    "7Networks_*_Default_PFC_*": {
        "dominant": "music",
        "video_weight": 0.35,
        "music_weight": 0.65,
        "confidence": 0.75
    },

    # ===== DEFAULT MODE – TEMPORAL =====
    "7Networks_*_Default_Temp_*": {
        "dominant": "music",
        "video_weight": 0.40,
        "music_weight": 0.60,
        "confidence": 0.55
    },

    # ===== DEFAULT MODE – PARIETAL =====
    "7Networks_*_Default_Par_*": {
        "dominant": "balanced",
        "video_weight": 0.45,
        "music_weight": 0.55,
        "confidence": 0.50
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "music",
        "video_weight": 0.35,
        "music_weight": 0.55,
        "confidence": 0.55
    }
}

regions_multimodal_class_claude_opus45 = {
    # ===== VISUAL NETWORK =====
    # Primary and secondary visual cortex - strictly unimodal
    # Literature: Felleman & Van Essen (1991), Ungerleider & Haxby (1994)
    "7Networks_*_Vis_*": {
        "dominant": "unimodal",
        "multimodal_weight": 0.10,
        "unimodal_weight": 0.90,
        "confidence": 0.95
    },

    # ===== SOMATOMOTOR =====
    # Primary motor and somatosensory - mostly unimodal
    # But some rhythm/beat processing involves motor areas
    # Literature: Zatorre et al. (2007), Grahn & Brett (2007)
    "7Networks_*_SomMot_*": {
        "dominant": "unimodal",
        "multimodal_weight": 0.25,
        "unimodal_weight": 0.75,
        "confidence": 0.80
    },

    # ===== DORSAL ATTENTION =====
    # Contains Intraparietal Sulcus (IPS) - known multimodal area
    # Spatial attention across modalities
    # Literature: Macaluso & Driver (2005), Shomstein & Yantis (2004)
    "7Networks_*_DorsAttn_Post_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.60,
        "unimodal_weight": 0.40,
        "confidence": 0.75
    },
    "7Networks_*_DorsAttn_FEF_*": {
        "dominant": "unimodal",
        "multimodal_weight": 0.30,
        "unimodal_weight": 0.70,
        "confidence": 0.70
    },
    "7Networks_*_DorsAttn_PrCv_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.50,
        "unimodal_weight": 0.50,
        "confidence": 0.65
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    # Contains Insula and TPJ - CLASSIC multimodal integration areas!
    # Literature: Calvert (2001), Beauchamp et al. (2004), Bushara et al. (2001)
    "7Networks_*_SalVentAttn_Ins_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.85,
        "unimodal_weight": 0.15,
        "confidence": 0.90
    },
    "7Networks_*_SalVentAttn_TempOccPar_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.80,
        "unimodal_weight": 0.20,
        "confidence": 0.85
    },
    "7Networks_*_SalVentAttn_FrOper_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.70,
        "unimodal_weight": 0.30,
        "confidence": 0.75
    },
    "7Networks_*_SalVentAttn_Med_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.65,
        "unimodal_weight": 0.35,
        "confidence": 0.70
    },
    "7Networks_*_SalVentAttn_ParOper_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.75,
        "unimodal_weight": 0.25,
        "confidence": 0.80
    },
    "7Networks_*_SalVentAttn_PrC_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.55,
        "unimodal_weight": 0.45,
        "confidence": 0.65
    },

    # ===== LIMBIC =====
    # Emotional processing - integrates across modalities for emotional salience
    # Literature: Ethofer et al. (2006), Klasen et al. (2011)
    "7Networks_*_Limbic_OFC_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.60,
        "unimodal_weight": 0.40,
        "confidence": 0.70
    },
    "7Networks_*_Limbic_TempPole_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.65,
        "unimodal_weight": 0.35,
        "confidence": 0.75
    },

    # ===== CONTROL / FRONTOPARIETAL =====
    # Executive control - integrates information across modalities
    # Literature: Miller & Cohen (2001), Duncan (2010)
    "7Networks_*_Cont_PFCl_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.55,
        "unimodal_weight": 0.45,
        "confidence": 0.65
    },
    "7Networks_*_Cont_PFCv_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.50,
        "unimodal_weight": 0.50,
        "confidence": 0.60
    },
    "7Networks_*_Cont_Par_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.55,
        "unimodal_weight": 0.45,
        "confidence": 0.65
    },
    "7Networks_*_Cont_Cing_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.50,
        "unimodal_weight": 0.50,
        "confidence": 0.60
    },
    "7Networks_*_Cont_PFCmp_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.45,
        "unimodal_weight": 0.55,
        "confidence": 0.55
    },
    "7Networks_*_Cont_pCun_*": {
        "dominant": "unimodal",
        "multimodal_weight": 0.40,
        "unimodal_weight": 0.60,
        "confidence": 0.60
    },

    # ===== DEFAULT MODE – PFC =====
    # Medial prefrontal - self-referential, semantic processing
    # Some multimodal integration for meaning
    # Literature: Binder et al. (2009), Andrews-Hanna et al. (2010)
    "7Networks_*_Default_PFC_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.50,
        "unimodal_weight": 0.50,
        "confidence": 0.60
    },

    # ===== DEFAULT MODE – TEMPORAL =====
    # May include regions near STS - potential multimodal
    # Literature: Beauchamp (2005) - STS is multimodal hub
    "7Networks_*_Default_Temp_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.55,
        "unimodal_weight": 0.45,
        "confidence": 0.65
    },

    # ===== DEFAULT MODE – PARIETAL =====
    # Angular gyrus - semantic integration across modalities
    # Literature: Seghier (2013), Price (2010)
    "7Networks_*_Default_Par_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.60,
        "unimodal_weight": 0.40,
        "confidence": 0.70
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    # Higher-order integration, episodic memory
    # Literature: Cavanna & Trimble (2006)
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "multimodal",
        "multimodal_weight": 0.50,
        "unimodal_weight": 0.50,
        "confidence": 0.60
    }
}


regions_audiovisual_interaction_analysis_gemini = {
    # ===== VISUAL NETWORK =====
    # Primary Visual Cortex (V1/V2)
    # Dynamics: Generally independent, but can suffer cross-modal suppression if auditory attention is high.
    # Literature: Laurienti et al. (2002) - Cross-modal deactivation in sensory specific cortices.
    "7Networks_Vis": {
        "dominant_effect": "independent_or_suppression",
        "enhancement_prob": 0.10,
        "suppression_prob": 0.60,
        "independent_prob": 0.30,
        "confidence": 0.90
    },

    # ===== SOMATOMOTOR =====
    # Motor/Auditory Periphery
    # Dynamics: Music drives rhythm/beat. Visuals of movement (dancing) can enhance this via Action Observation.
    # Literature: Haslinger et al. (2005) - Audio-visual observation of actions enhances motor excitability.
    "7Networks_*_SomMot_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.65,
        "suppression_prob": 0.10,
        "independent_prob": 0.25,
        "confidence": 0.80
    },

    # ===== DORSAL ATTENTION =====
    # Intraparietal Sulcus (IPS) / FEF
    # Dynamics: Top-down resource allocation. Dual-stream requires higher attentional load -> Higher activation.
    # Literature: Santangelo et al. (2008) - Multisensory integration reduces attentional blink, engaging DAN.
    "7Networks_*_DorsAttn_Post_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.75,
        "suppression_prob": 0.05,
        "independent_prob": 0.20,
        "confidence": 0.85
    },
    "7Networks_*_DorsAttn_FEF_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.70,
        "suppression_prob": 0.10,
        "independent_prob": 0.20,
        "confidence": 0.75
    },
    "7Networks_*_DorsAttn_PrCv_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.60,
        "suppression_prob": 0.10,
        "independent_prob": 0.30,
        "confidence": 0.70
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    # Insula, Operculum, TPJ
    # Dynamics: The "Switchboard". Highly sensitive to multimodal coincidence. Strong enhancement/binding.
    # Literature: Downar et al. (2000), Bushara et al. (2003) - Detection of audio-visual synchrony.
    "7Networks_*_SalVentAttn_Ins_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.90,
        "suppression_prob": 0.05,
        "independent_prob": 0.05,
        "confidence": 0.95
    },
    "7Networks_*_SalVentAttn_TempOccPar_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.85,
        "suppression_prob": 0.05,
        "independent_prob": 0.10,
        "confidence": 0.90
    },
    "7Networks_*_SalVentAttn_FrOper_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.80,
        "suppression_prob": 0.10,
        "independent_prob": 0.10,
        "confidence": 0.85
    },
    "7Networks_*_SalVentAttn_Med_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.70,
        "suppression_prob": 0.10,
        "independent_prob": 0.20,
        "confidence": 0.75
    },
    "7Networks_*_SalVentAttn_ParOper_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.80,
        "suppression_prob": 0.10,
        "independent_prob": 0.10,
        "confidence": 0.80
    },
    "7Networks_*_SalVentAttn_PrC_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.65,
        "suppression_prob": 0.15,
        "independent_prob": 0.20,
        "confidence": 0.70
    },

    # ===== LIMBIC =====
    # Emotional Integration
    # Dynamics: "Affective Congruence". Music + Video = Stronger emotional response than either alone.
    # Literature: Baumgartner et al. (2006) - Combined stimuli increase activation in amygdala/limbic system.
    "7Networks_*_Limbic_OFC_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.85,
        "suppression_prob": 0.05,
        "independent_prob": 0.10,
        "confidence": 0.85
    },
    "7Networks_*_Limbic_TempPole_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.80,
        "suppression_prob": 0.05,
        "independent_prob": 0.15,
        "confidence": 0.80
    },

    # ===== CONTROL / FRONTOPARIETAL =====
    # Executive Control
    # Dynamics: Processing two streams increases cognitive load, requiring higher network engagement (Up-regulation).
    # Literature: Duncan (2010) - Multiple Demand Network activation increases with task complexity.
    "7Networks_*_Cont_PFCl_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.70,
        "suppression_prob": 0.05,
        "independent_prob": 0.25,
        "confidence": 0.75
    },
    "7Networks_*_Cont_PFCv_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.65,
        "suppression_prob": 0.05,
        "independent_prob": 0.30,
        "confidence": 0.70
    },
    "7Networks_*_Cont_Par_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.70,
        "suppression_prob": 0.05,
        "independent_prob": 0.25,
        "confidence": 0.75
    },
    "7Networks_*_Cont_Cing_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.65,
        "suppression_prob": 0.10,
        "independent_prob": 0.25,
        "confidence": 0.70
    },
    "7Networks_*_Cont_PFCmp_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.60,
        "suppression_prob": 0.10,
        "independent_prob": 0.30,
        "confidence": 0.65
    },
    "7Networks_*_Cont_pCun_*": {
        "dominant_effect": "independent",
        "enhancement_prob": 0.40,
        "suppression_prob": 0.20,
        "independent_prob": 0.40,
        "confidence": 0.60
    },

    # ===== DEFAULT MODE – PFC =====
    # Dynamics: DMN is Task-Negative. High engagement in Movie+Music usually leads to deeper SUPPRESSION.
    # However, if the content is highly narrative/autobiographical, it may engage.
    # General Rule: External Attention = Suppression of DMN.
    "7Networks_*_Default_PFC_*": {
        "dominant_effect": "suppression",
        "enhancement_prob": 0.20,
        "suppression_prob": 0.70,
        "independent_prob": 0.10,
        "confidence": 0.85
    },

    # ===== DEFAULT MODE – TEMPORAL (STS) =====
    # Superior Temporal Sulcus (STS)
    # Dynamics: The EXCEPTION in the DMN. STS is a known audiovisual integrator (lips+speech, motion+sound).
    # Literature: Calvert (2001), Beauchamp (2005) - Super-additivity in STS.
    "7Networks_*_Default_Temp_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.95,
        "suppression_prob": 0.05,
        "independent_prob": 0.00,
        "confidence": 0.95
    },

    # ===== DEFAULT MODE – PARIETAL =====
    # Angular Gyrus
    # Dynamics: Semantic integration. Likely enhancement if narrative matches lyrics/dialogue.
    "7Networks_*_Default_Par_*": {
        "dominant_effect": "enhancement",
        "enhancement_prob": 0.60,
        "suppression_prob": 0.20,
        "independent_prob": 0.20,
        "confidence": 0.70
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    # Dynamics: Generally suppressed during high external attention (watching a movie).
    "7Networks_*_Default_pCunPCC_*": {
        "dominant_effect": "suppression",
        "enhancement_prob": 0.25,
        "suppression_prob": 0.65,
        "independent_prob": 0.10,
        "confidence": 0.80
    }
}

regions_crossmodal_interaction_sonnet45 = {

    # ===== VISUAL NETWORK =====
    # Primary visual cortex - largely unaffected by auditory input
    # May show slight enhancement with congruent audio (driver et al., 2008)
    # Literature: Driver & Noesselt (2008), Laurienti et al. (2002)
    "7Networks_*_Vis_*": {
        "dominant": "neutral_to_enhancement",
        "suppression_prob": 0.15,
        "enhancement_prob": 0.25,
        "neutral_weight": 0.60,
        "confidence": 0.85,
        "notes": "Primary visual cortex shows minimal cross-modal interference; slight enhancement with rhythmic auditory stimulation"
    },

    # ===== AUDITORY NETWORK =====
    # Primary auditory cortex - processes music independently
    # Can show enhancement with visual rhythm (Besle et al., 2008)
    # Literature: Kayser et al. (2008), Besle et al. (2008), van Atteveldt et al. (2004)
    "Auditory_*_Primary_*": {
        "dominant": "neutral_to_enhancement",
        "suppression_prob": 0.10,
        "enhancement_prob": 0.30,
        "neutral_weight": 0.60,
        "confidence": 0.85,
        "notes": "A1/A2 relatively insulated but can show enhancement with congruent visual motion"
    },

    # ===== SOMATOMOTOR =====
    # Motor areas involved in rhythm/beat - STRONG enhancement with AV
    # Music + visual motion creates robust motor resonance
    # Literature: Chen et al. (2008), Grahn & Brett (2007), Zatorre et al. (2007)
    "7Networks_*_SomMot_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.05,
        "enhancement_prob": 0.70,
        "neutral_weight": 0.25,
        "confidence": 0.80,
        "notes": "Motor regions show strong facilitation with audiovisual rhythm; beat synchronization enhanced"
    },

    # ===== DORSAL ATTENTION - POSTERIOR =====
    # Spatial attention - can show COMPETITION under high load
    # Enhancement when stimuli are congruent spatially
    # Literature: Talsma et al. (2010), Santangelo & Macaluso (2013)
    "7Networks_*_DorsAttn_Post_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.40,
        "enhancement_prob": 0.40,
        "neutral_weight": 0.20,
        "confidence": 0.70,
        "notes": "IPS shows competition under high attentional load; enhancement with spatial occupancy"
    },

    # ===== DORSAL ATTENTION - FEF =====
    # Frontal eye fields - primarily visual attention
    # Moderate competition with complex auditory stimuli
    # Literature: Macaluso & Driver (2001)
    "7Networks_*_DorsAttn_FEF_*": {
        "dominant": "slight_suppression",
        "suppression_prob": 0.45,
        "enhancement_prob": 0.25,
        "neutral_weight": 0.30,
        "confidence": 0.65,
        "notes": "Visual attention resources may be diverted; moderate competition effects"
    },

    "7Networks_*_DorsAttn_PrCv_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.35,
        "enhancement_prob": 0.40,
        "neutral_weight": 0.25,
        "confidence": 0.60,
        "notes": "Precuneus shows mixed patterns depending on task demands"
    },

    # ===== SALIENCE / VENTRAL ATTENTION =====
    # INSULA - Classic SUPERADDITIVE responses!
    # Literature: Calvert et al. (2001), Bushara et al. (2003), Stevenson & James (2009)
    "7Networks_*_SalVentAttn_Ins_*": {
        "dominant": "strong_enhancement",
        "suppression_prob": 0.05,
        "enhancement_prob": 0.85,
        "neutral_weight": 0.10,
        "confidence": 0.95,
        "notes": "Anterior insula shows superadditive responses to AV stimuli; key integration hub"
    },

    # TEMPORAL-OCCIPITAL-PARIETAL JUNCTION (includes STS region)
    # STRONGEST enhancement area - the gold standard for AV integration
    # Literature: Beauchamp et al. (2004), Calvert et al. (2000), Wright et al. (2003)
    "7Networks_*_SalVentAttn_TempOccPar_*": {
        "dominant": "strong_enhancement",
        "suppression_prob": 0.05,
        "enhancement_prob": 0.90,
        "neutral_weight": 0.05,
        "confidence": 0.95,
        "notes": "STS/TPJ junction - strongest superadditive AV responses in literature; music+video optimal"
    },

    # Frontal Operculum - multisensory integration
    # Literature: Foxe & Schroeder (2005)
    "7Networks_*_SalVentAttn_FrOper_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.10,
        "enhancement_prob": 0.75,
        "neutral_weight": 0.15,
        "confidence": 0.80,
        "notes": "Ventral frontal regions show facilitation for AV processing"
    },

    "7Networks_*_SalVentAttn_Med_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.15,
        "enhancement_prob": 0.65,
        "neutral_weight": 0.20,
        "confidence": 0.70,
        "notes": "Medial salience regions integrate emotional AV content"
    },

    "7Networks_*_SalVentAttn_ParOper_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.10,
        "enhancement_prob": 0.70,
        "neutral_weight": 0.20,
        "confidence": 0.75,
        "notes": "Parietal operculum shows audiovisual integration benefits"
    },

    "7Networks_*_SalVentAttn_PrC_*": {
        "dominant": "mild_enhancement",
        "suppression_prob": 0.20,
        "enhancement_prob": 0.55,
        "neutral_weight": 0.25,
        "confidence": 0.65,
        "notes": "Precentral salience regions show moderate facilitation"
    },

    # ===== LIMBIC =====
    # Emotion processing - ENHANCED with multimodal emotional stimuli
    # Music + emotional video creates synergistic effects
    # Literature: Klasen et al. (2011), Ethofer et al. (2006), Baumgartner et al. (2006)
    "7Networks_*_Limbic_OFC_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.10,
        "enhancement_prob": 0.70,
        "neutral_weight": 0.20,
        "confidence": 0.80,
        "notes": "Orbitofrontal cortex integrates emotional value across modalities; music enhances video emotion"
    },

    # Temporal pole - semantic and emotional integration
    # Literature: Olson et al. (2007), Visser et al. (2010)
    "7Networks_*_Limbic_TempPole_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.10,
        "enhancement_prob": 0.75,
        "neutral_weight": 0.15,
        "confidence": 0.80,
        "notes": "Temporal pole shows enhanced response to congruent AV emotional content"
    },

    # ===== CONTROL / FRONTOPARIETAL =====
    # Executive control - can show SUPPRESSION under high cognitive load
    # But enhancement for integrated task demands
    # Literature: Uncapher & Rugg (2008), Nijboer et al. (2014), Loose et al. (2003)

    "7Networks_*_Cont_PFCl_*": {
        "dominant": "suppression_under_load",
        "suppression_prob": 0.55,
        "enhancement_prob": 0.25,
        "neutral_weight": 0.20,
        "confidence": 0.75,
        "notes": "Lateral PFC shows competition when dual-task demands are high; perceptual load theory applies"
    },

    "7Networks_*_Cont_PFCv_*": {
        "dominant": "suppression_under_load",
        "suppression_prob": 0.50,
        "enhancement_prob": 0.30,
        "neutral_weight": 0.20,
        "confidence": 0.70,
        "notes": "Ventral PFC may show resource competition with complex AV stimuli"
    },

    "7Networks_*_Cont_Par_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.45,
        "enhancement_prob": 0.35,
        "neutral_weight": 0.20,
        "confidence": 0.70,
        "notes": "Parietal control regions show mixed patterns; depends on task integration"
    },

    "7Networks_*_Cont_Cing_*": {
        "dominant": "slight_suppression",
        "suppression_prob": 0.45,
        "enhancement_prob": 0.30,
        "neutral_weight": 0.25,
        "confidence": 0.65,
        "notes": "Cingulate may signal conflict with competing AV demands"
    },

    "7Networks_*_Cont_PFCmp_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.40,
        "enhancement_prob": 0.35,
        "neutral_weight": 0.25,
        "confidence": 0.60,
        "notes": "Medial prefrontal control shows variable patterns"
    },

    "7Networks_*_Cont_pCun_*": {
        "dominant": "neutral_to_mild_suppression",
        "suppression_prob": 0.40,
        "enhancement_prob": 0.30,
        "neutral_weight": 0.30,
        "confidence": 0.60,
        "notes": "Posterior control regions may show mild competition effects"
    },

    # ===== DEFAULT MODE – PFC =====
    # DMN typically SUPPRESSED during external tasks
    # But can show enhancement for narrative/semantic integration
    # Literature: Yeshurun et al. (2017), Hasson et al. (2008), Naci et al. (2014)
    "7Networks_*_Default_PFC_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.35,
        "enhancement_prob": 0.45,
        "neutral_weight": 0.20,
        "confidence": 0.65,
        "notes": "mPFC can be suppressed by external tasks BUT enhanced when music+video create coherent narrative"
    },

    # ===== DEFAULT MODE – TEMPORAL =====
    # Includes regions near STS - can show ENHANCEMENT for stories
    # Literature: Lerner et al. (2011), Regev et al. (2013)
    "7Networks_*_Default_Temp_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.20,
        "enhancement_prob": 0.60,
        "neutral_weight": 0.20,
        "confidence": 0.75,
        "notes": "Temporal DMN shows enhanced activity for integrated AV narratives; music+movie synergy"
    },

    # ===== DEFAULT MODE – PARIETAL =====
    # Angular gyrus - semantic integration ENHANCED
    # Literature: Bonner et al. (2013), Seghier (2013)
    "7Networks_*_Default_Par_*": {
        "dominant": "enhancement",
        "suppression_prob": 0.15,
        "enhancement_prob": 0.65,
        "neutral_weight": 0.20,
        "confidence": 0.80,
        "notes": "Angular gyrus integrates meaning across modalities; music enhances semantic processing of video"
    },

    # ===== DEFAULT MODE – PCC / Precuneus =====
    # Can be suppressed by attention demands but enhanced by memory integration
    # Literature: Cavanna & Trimble (2006), Margulies et al. (2009)
    "7Networks_*_Default_pCunPCC_*": {
        "dominant": "context_dependent",
        "suppression_prob": 0.35,
        "enhancement_prob": 0.40,
        "neutral_weight": 0.25,
        "confidence": 0.65,
        "notes": "PCC/Precuneus shows suppression with high attentional load but enhancement for episodic AV integration"
    }

}

def roi_vis_prob(atlas_labels, regions_class_origin, regions_multimodal_class_origin, use_va=False):
    if not use_va:
        if regions_class_origin == 'chatgpt':
            regions_class = regions_class_chatgpt
        elif regions_class_origin == 'gemini':
            regions_class = regions_class_gemini
        elif regions_class_origin == 'chatgpt52':
            regions_class = regions_class_chatgpt52
        elif regions_class_origin == 'claude_opus45':
            regions_class = regions_class_claude_opus45
    else:
        # if regions_multimodal_class_origin == 'claude_opus45':
        #     regions_class = regions_multimodal_class_claude_opus45
        if regions_multimodal_class_origin == 'gemini':
            regions_class = regions_audiovisual_interaction_analysis_gemini
        elif regions_multimodal_class_origin == 'claude_sonnet45':
            regions_class = regions_crossmodal_interaction_sonnet45

    region_vis_prob = {}
    for label_bytes in atlas_labels:
        label = label_bytes.decode('utf-8')
        for pattern, info in regions_class.items():
            if fnmatch.fnmatch(label, pattern):
                if use_va:
                    prob_en = info["enhancement_prob"]
                    prob_su = info["suppression_prob"]
                    prob = prob_en / (prob_en + prob_su)
                else:
                    prob_v = info["video_weight"]
                    prob_a = info["music_weight"]
                    prob = prob_v / (prob_v + prob_a)
                region_vis_prob[label] = prob
                break
        if label not in region_vis_prob:
            region_vis_prob[label] = 0.0
    return region_vis_prob

BLOCKS = [
    (0, 20, 'va'), (20, 40, 'a'), (40, 60, 'v'),
    (60, 80, 'a'), (80, 100, 'va'), (100, 120, 'v'),
    (120, 140, 'va'), (140, 160, 'x'), (160, 180, 'v'),
    (180, 200, 'a'), (200, 220, 'x'), (220, 240, 'a'),
    (240, 260, 'va'), (260, 280, 'v'), (280, 300, 'va'),
    (300, 320, 'v'), (320, 340, 'a'), (340, 360, 'v'),
    (360, 380, 'a'), (380, 400, 'va'), (400, 420, 'v'),
    (420, 440, 'x'), (440, 450, 'v'), (450, 460, 'va')
]
