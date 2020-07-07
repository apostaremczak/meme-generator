const GeneratorEndpoint = "generate"

const categories = document.querySelectorAll(".category-image")
console.log(categories)
categories.forEach(category => {
    category.addEventListener("click", () => {
        fetch(`${GeneratorEndpoint}/${category.id}`)
            .then(res => res.json())
            .then(data => document.getElementById("result-image").src = data.result_image)
        window.scrollTo({ top: 0, behavior: 'smooth' })
    })
})
