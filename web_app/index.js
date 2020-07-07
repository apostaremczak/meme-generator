const GeneratorEndpoint = "https://server-placeholder.com/generate"

async function app() {
  const categories = document.querySelectorAll("category-image")
  categories.forEach(function (category) {
    category.addEventListener("click", function () {
      console.log(category.id)
    })
  })
}

app();
