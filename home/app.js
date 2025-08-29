// app.js

const translations = {
  en: {
    brand: "MIP Candy",
    "nav.features": "Features",
    "nav.quickstart": "Quick Start",
    "nav.arch": "Architecture",
    "nav.links": "Links",
    "hero.title": "A Candy for Medical Image Processing",
    "hero.subtitle": "Next-generation framework for medical imaging with out-of-the-box training, inference, evaluation, and rich integrations.",
    "cta.getStarted": "Get Started",
    "cta.docs": "Docs & Home",
    "features.title": "Key Features",
    "features.training.title": "Out-of-the-box Training Pipeline",
    "features.training.desc": "Complete training workflows for segmentation and other medical tasks.",
    "features.data.title": "Data Handling & Formats",
    "features.data.desc": "NIfTI/DICOM via SimpleITK, preprocessing, transforms, and visualization.",
    "features.arch.title": "Specialized Architectures",
    "features.arch.desc": "Pre-configured models and sliding-window inference for large volumes.",
    "features.metrics.title": "Medical Metrics",
    "features.metrics.desc": "Dice, IoU, precision/recall and evaluation helpers.",
    "features.exp.title": "Experiment Tracking",
    "features.exp.desc": "Integrations with Notion, WandB, and TensorBoard.",
    "features.viz.title": "2D/3D Visualization",
    "features.viz.desc": "From slice viewers to volume rendering (pyvista via optional extras).",
    "quick.title": "Quick Start",
    "quick.require": "Requirements",
    "quick.install": "Install",
    "arch.title": "Architecture",
    "arch.data": "I/O, datasets, transforms, visualization.",
    "arch.train": "Trainer / SlidingTrainer, metrics tracking.",
    "arch.infer": "Predictor and inference utilities.",
    "arch.eval": "Evaluator and medical metrics.",
    "arch.fe": "Notion/W&B integrations.",
    "arch.preset": "Pre-configured trainers for common tasks.",
    "links.title": "Links",
    "links.home": "Homepage",
    "links.docs": "Documentation",
    "links.contact": "Contact",
    "footer": "Apache-2.0 Licensed. © Project Neura."
  },
  zh: {
    brand: "MIP Candy",
    "nav.features": "特性",
    "nav.quickstart": "快速开始",
    "nav.arch": "架构",
    "nav.links": "链接",
    "hero.title": "面向医疗影像的下一代基础框架",
    "hero.subtitle": "开箱即用的训练 / 推理 / 评估与可视化，完善的生态集成，助力医疗影像工作流程落地。",
    "cta.getStarted": "开始使用",
    "cta.docs": "文档与主页",
    "features.title": "核心特性",
    "features.training.title": "开箱即用的训练管线",
    "features.training.desc": "针对分割等医疗任务提供完整训练流程。",
    "features.data.title": "数据处理与格式",
    "features.data.desc": "基于 SimpleITK 的 NIfTI/DICOM 支持，预处理、变换与可视化。",
    "features.arch.title": "专用网络与推理",
    "features.arch.desc": "预配置模型，大体积三维数据滑窗推理。",
    "features.metrics.title": "医疗指标",
    "features.metrics.desc": "Dice、IoU、精确率/召回率与评估工具。",
    "features.exp.title": "实验跟踪",
    "features.exp.desc": "集成 Notion、WandB 与 TensorBoard。",
    "features.viz.title": "2D/3D 可视化",
    "features.viz.desc": "从切片查看到体渲染（可选 extras 启用 pyvista）。",
    "quick.title": "快速开始",
    "quick.require": "运行环境",
    "quick.install": "安装",
    "arch.title": "架构",
    "arch.data": "I/O、数据集、变换与可视化。",
    "arch.train": "Trainer / SlidingTrainer，指标记录。",
    "arch.infer": "Predictor 与推理工具。",
    "arch.eval": "Evaluator 与医疗指标。",
    "arch.fe": "Notion / W&B 集成。",
    "arch.preset": "常见任务的预配置训练器。",
    "links.title": "链接",
    "links.home": "主页",
    "links.docs": "文档",
    "links.contact": "联系",
    "footer": "Apache-2.0 许可。© Project Neura。"
  },
  fr: {
    brand: "MIP Candy",
    "nav.features": "Fonctionnalités",
    "nav.quickstart": "Démarrage Rapide",
    "nav.arch": "Architecture",
    "nav.links": "Liens",
    "hero.title": "Une Friandise pour le Traitement d'Images Médicales",
    "hero.subtitle": "Framework de nouvelle génération pour l'imagerie médicale avec entraînement, inférence, évaluation prêts à l'emploi et intégrations riches.",
    "cta.getStarted": "Commencer",
    "cta.docs": "Docs & Accueil",
    "features.title": "Fonctionnalités Clés",
    "features.training.title": "Pipeline d'Entraînement Prêt à l'Emploi",
    "features.training.desc": "Workflows d'entraînement complets pour la segmentation et autres tâches médicales.",
    "features.data.title": "Gestion des Données et Formats",
    "features.data.desc": "Support NIfTI/DICOM via SimpleITK, prétraitement, transformations et visualisation.",
    "features.arch.title": "Architectures Spécialisées",
    "features.arch.desc": "Modèles préconfigurés et inférence par fenêtre glissante pour gros volumes.",
    "features.metrics.title": "Métriques Médicales",
    "features.metrics.desc": "Dice, IoU, précision/rappel et outils d'évaluation.",
    "features.exp.title": "Suivi d'Expériences",
    "features.exp.desc": "Intégrations avec Notion, WandB et TensorBoard.",
    "features.viz.title": "Visualisation 2D/3D",
    "features.viz.desc": "Des visualiseurs de coupes au rendu volumique (pyvista via extras optionnels).",
    "quick.title": "Démarrage Rapide",
    "quick.require": "Prérequis",
    "quick.install": "Installation",
    "arch.title": "Architecture",
    "arch.data": "I/O, jeux de données, transformations, visualisation.",
    "arch.train": "Trainer / SlidingTrainer, suivi des métriques.",
    "arch.infer": "Predictor et utilitaires d'inférence.",
    "arch.eval": "Evaluator et métriques médicales.",
    "arch.fe": "Intégrations Notion/W&B.",
    "arch.preset": "Entraîneurs préconfigurés pour tâches communes.",
    "links.title": "Liens",
    "links.home": "Accueil",
    "links.docs": "Documentation",
    "links.contact": "Contact",
    "footer": "Licence Apache-2.0. © Project Neura."
  }
};

// Apply translations
function applyI18n(lang) {
  const dict = translations[lang] || translations.en;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) el.textContent = dict[key];
  });
  // Optional: set <html lang="">
  document.documentElement.setAttribute("lang", lang);
}

// Language detection & toggle
const storageKey = "mipcandy.lang";
function getInitialLang() {
  const saved = localStorage.getItem(storageKey);
  if (saved) return saved;
  const nav = (navigator.language || "").toLowerCase();
  if (nav.startsWith("zh")) return "zh";
  if (nav.startsWith("fr")) return "fr";
  return "en";
}

function setLanguage(lang) {
  localStorage.setItem(storageKey, lang);
  applyI18n(lang);
}

document.addEventListener("DOMContentLoaded", () => {
  const initial = getInitialLang();
  applyI18n(initial);

  const btn = document.getElementById("langToggle");
  if (btn) {
    function updateButtonText(lang) {
      const texts = {
        "en": "EN / 中文 / FR",
        "zh": "中文 / FR / EN", 
        "fr": "FR / EN / 中文"
      };
      btn.textContent = texts[lang] || texts["en"];
    }
    
    updateButtonText(initial);
    
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("lang") || "en";
      const cycle = { "en": "zh", "zh": "fr", "fr": "en" };
      const next = cycle[current] || "en";
      setLanguage(next);
      updateButtonText(next);
    });
  }
});
