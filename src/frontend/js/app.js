"use strict";

/**
 * Telecom Retention Intelligence
 *
 * The frontend deliberately contains no ML logic.
 * It validates basic form input, communicates with FastAPI,
 * and renders the API response.
 */

const API_ENDPOINTS = Object.freeze({
  health: "/health",
  prediction: "/predictions/churn"
});

const elements = {
  form: document.querySelector("#risk-form"),
  assessButton: document.querySelector("#assess-button"),
  retryButton: document.querySelector("#retry-button"),

  apiStatusDot: document.querySelector("#api-status-dot"),
  apiStatusText: document.querySelector("#api-status-text"),
  apiWarning: document.querySelector("#api-warning"),

  emptyState: document.querySelector("#empty-state"),
  loadingState: document.querySelector("#loading-state"),
  predictionResults: document.querySelector("#prediction-results"),
  resultError: document.querySelector("#result-error"),

  formError: document.querySelector("#form-error"),
  liveRegion: document.querySelector("#live-region"),

  probabilityValue: document.querySelector("#probability-value"),
  probabilityFill: document.querySelector("#probability-fill"),
  probabilityTrack: document.querySelector(".probability-track"),

  riskStatus: document.querySelector("#risk-status"),
  riskBand: document.querySelector("#risk-band"),
  retentionFlag: document.querySelector("#retention-flag"),

  revenueAtRisk: document.querySelector("#revenue-at-risk"),
  priorityTier: document.querySelector("#priority-tier"),
  retentionUrgency: document.querySelector("#retention-urgency"),
  retentionAction: document.querySelector("#retention-action")
};

const fieldDefinitions = [
  {
    name: "CRM_PID_Value_Segment",
    id: "crm-segment",
    type: "category"
  },
  {
    name: "EffectiveSegment",
    id: "effective-segment",
    type: "category"
  },
  {
    name: "Active_subscribers",
    id: "active-subscribers",
    type: "nonNegativeInteger"
  },
  {
    name: "Not_Active_subscribers",
    id: "not-active-subscribers",
    type: "nonNegativeInteger"
  },
  {
    name: "Suspended_subscribers",
    id: "suspended-subscribers",
    type: "nonNegativeInteger"
  },
  {
    name: "Total_SUBs",
    id: "total-subscribers",
    type: "positiveInteger"
  },
  {
    name: "AvgMobileRevenue",
    id: "mobile-revenue",
    type: "nonNegativeNumber"
  },
  {
    name: "AvgFIXRevenue",
    id: "fixed-revenue",
    type: "nonNegativeNumber"
  },
  {
    name: "ARPU",
    id: "arpu",
    type: "nonNegativeNumber"
  },
  {
    name: "TotalRevenue",
    id: "total-revenue",
    type: "nonNegativeNumber"
  }
];

let lastSuccessfulPayload = null;

document.addEventListener("DOMContentLoaded", initializeApplication);

function initializeApplication() {
  bindEvents();
  checkApiHealth();
}

function bindEvents() {
  elements.form.addEventListener("submit", handleFormSubmit);

  elements.retryButton.addEventListener("click", () => {
    if (lastSuccessfulPayload) {
      submitPrediction(lastSuccessfulPayload);
      return;
    }

    elements.form.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  });

  elements.form.addEventListener("input", handleFieldInput);
  elements.form.addEventListener("change", handleFieldInput);
}

async function checkApiHealth() {
  setApiStatus("checking");

  try {
    const response = await fetch(API_ENDPOINTS.health, {
      method: "GET",
      headers: {
        Accept: "application/json"
      },
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error("Health endpoint returned a non-success status.");
    }

    const data = await parseJsonResponse(response);

    if (data?.status !== "healthy") {
      throw new Error("Health endpoint did not report a healthy status.");
    }

    setApiStatus("online");
  } catch (error) {
    setApiStatus("offline");
  }
}

function setApiStatus(status) {
  elements.apiStatusDot.classList.remove(
    "status-dot--checking",
    "status-dot--online",
    "status-dot--offline"
  );

  if (status === "online") {
    elements.apiStatusDot.classList.add("status-dot--online");
    elements.apiStatusText.textContent = "API Online";
    elements.apiWarning.hidden = true;
    return;
  }

  if (status === "offline") {
    elements.apiStatusDot.classList.add("status-dot--offline");
    elements.apiStatusText.textContent = "API Unavailable";
    elements.apiWarning.hidden = false;
    return;
  }

  elements.apiStatusDot.classList.add("status-dot--checking");
  elements.apiStatusText.textContent = "Checking API...";
}

async function handleFormSubmit(event) {
  event.preventDefault();

  clearFormError();

  const validation = validateForm();

  if (!validation.valid) {
    showFormError("Please correct the highlighted fields.");
    focusFirstInvalidField();
    announce("Please correct the highlighted fields.");
    return;
  }

  const payload = getFormData();

  lastSuccessfulPayload = payload;

  await submitPrediction(payload);
}

function validateForm() {
  let isValid = true;

  clearFieldErrors();

  for (const field of fieldDefinitions) {
    const input = document.getElementById(field.id);

    if (!input) {
      continue;
    }

    const rawValue = input.value.trim();
    let errorMessage = "";

    if (!rawValue) {
      errorMessage = "This field is required.";
    } else if (field.type === "category") {
      errorMessage = validateCategory(rawValue);
    } else if (field.type === "nonNegativeInteger") {
      errorMessage = validateNonNegativeInteger(rawValue);
    } else if (field.type === "positiveInteger") {
      errorMessage = validatePositiveInteger(rawValue);
    } else if (field.type === "nonNegativeNumber") {
      errorMessage = validateNonNegativeNumber(rawValue);
    }

    if (errorMessage) {
      setFieldError(input, errorMessage);
      isValid = false;
    }
  }

  return {
    valid: isValid
  };
}

function validateCategory(value) {
  return value.length === 0
    ? "Please select a segment."
    : "";
}

function validateNonNegativeInteger(value) {
  const number = Number(value);

  if (!Number.isInteger(number)) {
    return "Enter a whole number.";
  }

  if (number < 0) {
    return "Value cannot be negative.";
  }

  return "";
}

function validatePositiveInteger(value) {
  const number = Number(value);

  if (!Number.isInteger(number)) {
    return "Enter a whole number.";
  }

  if (number <= 0) {
    return "Total subscribers must be greater than 0.";
  }

  return "";
}

function validateNonNegativeNumber(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "Enter a valid number.";
  }

  if (number < 0) {
    return "Value cannot be negative.";
  }

  return "";
}

function getFormData() {
  return {
    CRM_PID_Value_Segment: getStringValue("crm-segment"),
    EffectiveSegment: getStringValue("effective-segment"),

    Active_subscribers: getIntegerValue("active-subscribers"),
    Not_Active_subscribers: getIntegerValue("not-active-subscribers"),
    Suspended_subscribers: getIntegerValue("suspended-subscribers"),
    Total_SUBs: getIntegerValue("total-subscribers"),

    AvgMobileRevenue: getNumberValue("mobile-revenue"),
    AvgFIXRevenue: getNumberValue("fixed-revenue"),
    ARPU: getNumberValue("arpu"),
    TotalRevenue: getNumberValue("total-revenue")
  };
}

function getStringValue(id) {
  return document.getElementById(id).value.trim();
}

function getIntegerValue(id) {
  return Number.parseInt(document.getElementById(id).value, 10);
}

function getNumberValue(id) {
  return Number.parseFloat(document.getElementById(id).value);
}

async function submitPrediction(payload) {
  setLoadingState(true);
  resetResults();

  announce("Assessing customer risk...");

  try {
    const response = await fetch(API_ENDPOINTS.prediction, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      await safelyConsumeErrorResponse(response);
      throw new Error("Prediction request failed.");
    }

    const result = await parseJsonResponse(response);

    validatePredictionResponse(result);
    renderPrediction(result);

    announce("Customer risk assessment completed.");
  } catch (error) {
    renderPredictionError();
    announce("Unable to assess customer risk. Please try again.");
  } finally {
    setLoadingState(false);
  }
}

async function parseJsonResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  if (!contentType.includes("application/json")) {
    throw new Error("Server returned an unexpected response format.");
  }

  return response.json();
}

async function safelyConsumeErrorResponse(response) {
  try {
    await response.text();
  } catch {
    // Deliberately ignore response parsing failures.
  }
}

function validatePredictionResponse(result) {
  if (!result || typeof result !== "object") {
    throw new Error("Prediction response is invalid.");
  }

  const requiredFields = [
    "Churn_Probability",
    "Retention_Flag",
    "Risk_Band",
    "Revenue_at_Risk",
    "Priority_Tier",
    "Retention_Action",
    "Retention_Urgency"
  ];

  const missingField = requiredFields.find(
    (field) => !(field in result)
  );

  if (missingField) {
    throw new Error("Prediction response is incomplete.");
  }

  if (!Number.isFinite(Number(result.Churn_Probability))) {
    throw new Error("Prediction probability is invalid.");
  }

  if (!Number.isFinite(Number(result.Revenue_at_Risk))) {
    throw new Error("Revenue risk value is invalid.");
  }

  if (typeof result.Retention_Flag !== "boolean") {
    throw new Error("Retention flag is invalid.");
  }
}

function renderPrediction(result) {
  const probability = normalizeProbability(result.Churn_Probability);
  const risk = normalizeRiskBand(result.Risk_Band);

  const formattedProbability = formatProbability(probability);
  const formattedRevenue = formatCurrency(result.Revenue_at_Risk);

  elements.probabilityValue.textContent = formattedProbability;

  elements.probabilityFill.style.width = `${probability}%`;
  elements.probabilityFill.dataset.risk = risk;

  elements.probabilityTrack.setAttribute(
    "aria-valuenow",
    String(Math.round(probability * 100) / 100)
  );

  elements.probabilityTrack.setAttribute(
    "aria-valuetext",
    formattedProbability
  );

  elements.riskStatus.textContent = formatDisplayText(result.Risk_Band);
  elements.riskStatus.dataset.risk = risk;

  elements.riskBand.textContent = formatDisplayText(result.Risk_Band);

  elements.retentionFlag.textContent = result.Retention_Flag
    ? "Action Required"
    : "Monitor";

  elements.retentionFlag.dataset.flag = result.Retention_Flag
    ? "action"
    : "monitor";

  elements.revenueAtRisk.textContent = formattedRevenue;
  elements.priorityTier.textContent = formatDisplayText(
    result.Priority_Tier
  );
  elements.retentionUrgency.textContent = formatDisplayText(
    result.Retention_Urgency
  );
  elements.retentionAction.textContent = String(
    result.Retention_Action
  );

  elements.emptyState.hidden = true;
  elements.loadingState.hidden = true;
  elements.resultError.hidden = true;
  elements.predictionResults.hidden = false;
}

function normalizeProbability(value) {
  const numericValue = Number(value);

  /*
   * FastAPI's documented example returns 0.079356.
   * The API contract describes this as a probability, so convert
   * the 0-1 probability to a percentage for presentation.
   */
  return Math.min(100, Math.max(0, numericValue * 100));
}

function normalizeRiskBand(value) {
  const normalized = String(value).trim().toLowerCase();

  if (normalized === "low") {
    return "low";
  }

  if (
    normalized === "moderate" ||
    normalized === "medium"
  ) {
    return "moderate";
  }

  if (normalized === "high") {
    return "high";
  }

  return "unknown";
}

function formatProbability(value) {
  return `${value.toFixed(2)}%`;
}

function formatCurrency(value) {
  const number = Number(value);

  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(number);
}

function formatDisplayText(value) {
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function setLoadingState(isLoading) {
  elements.assessButton.disabled = isLoading;
  elements.assessButton.classList.toggle("is-loading", isLoading);
  elements.assessButton.setAttribute("aria-busy", String(isLoading));

  const label = elements.assessButton.querySelector(".button__label");

  if (label) {
    label.textContent = isLoading
      ? "Assessing..."
      : "Assess Customer Risk";
  }

  elements.loadingState.hidden = !isLoading;

  if (isLoading) {
    elements.emptyState.hidden = true;
    elements.predictionResults.hidden = true;
    elements.resultError.hidden = true;
  }
}

function resetResults() {
  elements.emptyState.hidden = true;
  elements.loadingState.hidden = false;
  elements.predictionResults.hidden = true;
  elements.resultError.hidden = true;
}

function renderPredictionError() {
  elements.emptyState.hidden = true;
  elements.loadingState.hidden = true;
  elements.predictionResults.hidden = true;
  elements.resultError.hidden = false;
}

function handleFieldInput(event) {
  const input = event.target;

  if (!input.matches("input, select")) {
    return;
  }

  if (input.getAttribute("aria-invalid") !== "true") {
    return;
  }

  clearFieldError(input);
  clearFormError();
}

function setFieldError(input, message) {
  input.setAttribute("aria-invalid", "true");

  const errorElement = document.getElementById(
    `${input.id}-error`
  );

  if (errorElement) {
    errorElement.textContent = message;
  }
}

function clearFieldError(input) {
  input.removeAttribute("aria-invalid");

  const errorElement = document.getElementById(
    `${input.id}-error`
  );

  if (errorElement) {
    errorElement.textContent = "";
  }
}

function clearFieldErrors() {
  for (const field of fieldDefinitions) {
    const input = document.getElementById(field.id);

    if (input) {
      clearFieldError(input);
    }
  }
}

function clearFormError() {
  elements.formError.hidden = true;
  elements.formError.textContent = "";
}

function showFormError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function focusFirstInvalidField() {
  const invalidField = elements.form.querySelector(
    '[aria-invalid="true"]'
  );

  invalidField?.focus();
}

function announce(message) {
  elements.liveRegion.textContent = "";

  window.setTimeout(() => {
    elements.liveRegion.textContent = message;
  }, 50);
}