const GeneratorEndpoint = "generate"

const categories = document.querySelectorAll(".category-image")
console.log(categories)
categories.forEach(category => {
    category.addEventListener("click", () => {
        console.log(`Clicked on ${category.id}`)

        fetch(`${GeneratorEndpoint}/${category.id}`)
            .then(res => {
                console.log(res.json())
            })
    })
})