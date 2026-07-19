(() => {
  const picker = document.querySelector("[data-map-picker]");
  if (!picker) return;

  const radios = [...document.querySelectorAll('input[name="best_of"]')];
  const slots = [...picker.querySelectorAll("[data-map-slot]")];
  const selects = [...picker.querySelectorAll("[data-map-select]")];
  const pickSelects = [...picker.querySelectorAll("[data-map-pick]")];
  const autoButton = picker.querySelector("[data-map-auto]");

  const selectedBestOf = () => {
    const checked = radios.find((radio) => radio.checked);
    return Number(checked?.value || 3);
  };

  const syncUniqueOptions = () => {
    const chosen = new Set(selects.map((select) => select.value).filter(Boolean));
    selects.forEach((select) => {
      [...select.options].forEach((option) => {
        option.disabled = Boolean(option.value && option.value !== select.value && chosen.has(option.value));
      });
    });
  };

  const syncSlots = () => {
    const bestOf = selectedBestOf();
    slots.forEach((slot) => {
      const active = Number(slot.dataset.mapSlot) <= bestOf;
      slot.hidden = !active;
      const slotSelects = [...slot.querySelectorAll("select")];
      slotSelects.forEach((select) => {
        select.disabled = !active;
        if (!active) select.value = "";
      });
    });
    pickSelects.forEach((pickSelect, index) => {
      const mapSelected = Boolean(selects[index]?.value);
      pickSelect.disabled = pickSelect.closest("[data-map-slot]")?.hidden || !mapSelected;
      if (!mapSelected) pickSelect.value = "";
    });
    syncUniqueOptions();
  };

  radios.forEach((radio) => radio.addEventListener("change", syncSlots));
  selects.forEach((select) => select.addEventListener("change", syncSlots));
  autoButton?.addEventListener("click", () => {
    selects.forEach((select) => {
      select.value = "";
    });
    pickSelects.forEach((select) => {
      select.value = "";
    });
    syncSlots();
  });

  syncSlots();
})();

(() => {
  const status = document.querySelector("[data-job-status]");
  if (!status) return;

  const initialState = status.dataset.state;
  if (!["queued", "running"].includes(initialState)) return;

  const stateNode = status.querySelector("[data-job-state]");
  const messageNode = status.querySelector("[data-job-message]");
  const progressNode = status.querySelector("[data-job-progress]");

  const titleCase = (value) =>
    String(value || "")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());

  const poll = async () => {
    try {
      const response = await fetch("/api/job", { cache: "no-store" });
      if (!response.ok) throw new Error("Job status request failed");
      const job = await response.json();
      status.dataset.state = job.status || "";
      stateNode.textContent = titleCase(job.status);
      messageNode.textContent = job.message || "";
      progressNode.style.width = `${Math.max(0, Math.min(1, Number(job.progress || 0))) * 100}%`;
      if (["queued", "running"].includes(job.status)) {
        window.setTimeout(poll, 2000);
      } else {
        window.location.reload();
      }
    } catch (_error) {
      window.setTimeout(poll, 4000);
    }
  };

  window.setTimeout(poll, 750);
})();
