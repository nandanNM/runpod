async function askRunPod(question) {
  const url = "https://api.runpod.ai/v2/dfarvinue86ajr/runsync";

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer <api_key>"
      },
      body: JSON.stringify({
        input: {
          question
        }
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const result = await response.json();

    // Return only when the request has fully completed
    return result;
  } catch (error) {
    console.error("RunPod request failed:", error);
    throw error;
  }
}

// Usage
(async () => {
  const response = await askRunPod(
    "what is serverless GPU and what does it matter ?"
  );

  console.log(response);
})();