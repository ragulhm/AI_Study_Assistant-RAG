document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("askForm");
    const loading = document.getElementById("loading");

    if (form) {
        form.addEventListener("submit", function () {
            loading.style.display = "block";
        });
    }
});