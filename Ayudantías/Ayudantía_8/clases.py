from pydantic import BaseModel, Field
from typing import List


class Poem(BaseModel):
    title: str = Field(
        description="Title of the poem",
        examples=[
            "Yo te he nombrado reina.",
            "La tierra verde se ha entregado",
            "Ode to a Nightingale"
        ]
    )
    author: str = Field(
        description="Author of the poem",
        examples=["Pablo Neruda", "Unknown", "John Keats"]
    )
    content: str = Field(
        description="Full content of the poem",
        examples=[
            "Yo te he nombrado reina.\n\nHay más altas que tú, más altas.\nHay más puras que tú, más puras.\nHay más bellas que tú, hay más bellas.\n\nPero tú eres la reina.\n\nCuando vas por las calles\nnadie  te  reconoce.\nNadie ve tu corona de cristal, nadie mira\nLa alfombra de oro rojo\nque pisas donde pasas,\nla alfombra que no existe.\n\nY cuando asomas\nsuenan todos los ríos\nde mi cuerpo, sacuden\nel cielo las campanas,\ny un himno llena el mundo.",
            "La tierra verde se ha entregado\na todo lo amarillo, oro, cosechas,\nterrones, hojas, grano,\n\npero cuando el otoño\n \nse levanta\ncon su estandarte extenso\neres tú la que veo,\nes para mi tu cabellera\nla que reparte las espigas.",
            "My heart aches, and a drowsy numbness pains\nMy sense, as though of hemlock I had drunk,"
        ]
    )



class PoemExtractionOutput(BaseModel):
    poemas_encontrados: List[Poem] = Field(
        description="List of all extracted poems",
        examples=[
            [ 
                {
                    "title": "Yo te he nombrado reina.",
                    "author": "Unknown",
                    "content": "Yo te he nombrado reina.\n\nHay más altas que tú, más altas.\n..."
                },
                {
                    "title": "La tierra verde se ha entregado",
                    "author": "Unknown",
                    "content": "La tierra verde se ha entregado\na todo lo amarillo, oro, cosechas,\n..."
                }
            ]
        ]
    )

